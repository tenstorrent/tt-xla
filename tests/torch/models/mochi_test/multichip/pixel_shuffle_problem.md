# Pixel Shuffle: The DRAM OOM Bottleneck in Mochi Decoder

## Table of Contents

1. [What Is Pixel Shuffle?](#1-what-is-pixel-shuffle)
2. [Where It Appears in the Mochi Decoder](#2-where-it-appears-in-the-mochi-decoder)
3. [The Problem: Tile Alignment on Small Trailing Dimensions](#3-the-problem-tile-alignment-on-small-trailing-dimensions)
4. [Verified Impact: tt-swiss Memory Report](#4-verified-impact-tt-swiss-memory-report)
5. [Why the Decoder Cannot Fit in 128 GB (4x32 GB)](#5-why-the-decoder-cannot-fit-in-128-gb-4x32-gb)
6. [Options for Fixing It](#6-options-for-fixing-it)

---

## 1. What Is Pixel Shuffle?

**Pixel shuffle** (also called **sub-pixel convolution** or **depth-to-space**) is an
upsampling technique that trades channel depth for spatial resolution. Instead of using
transposed convolutions or bilinear interpolation to increase spatial size, pixel shuffle
rearranges elements from the channel dimension into the spatial dimensions.

For 2D images, PyTorch provides `nn.PixelShuffle(upscale_factor)`. For 3D video
(temporal + height + width), there is no built-in op -- Mochi implements its own
**depth-to-space-time** using three standard PyTorch ops:

```
reshape (5D -> 8D)  -->  permute (interleave)  -->  reshape (8D -> 5D)
```

### Concrete example (2D, simplified)

Given a `[1, 12, 2, 2]` tensor (12 channels, 2x2 spatial), pixel shuffle with
`upscale_factor=2` produces a `[1, 3, 4, 4]` tensor (3 channels, 4x4 spatial):

```
Step 1: reshape [1, 12, 2, 2] -> [1, 3, 2, 2, 2, 2]
            factor 12 = 3 * 2(sh) * 2(sw)

Step 2: permute(0, 1, 2, 4, 3, 5) -> [1, 3, 2, 2, 2, 2]
            interleave spatial dims with expansion factors

Step 3: reshape [1, 3, 2, 2, 2, 2] -> [1, 3, 4, 4]
            merge (H, sh) -> H*sh and (W, sw) -> W*sw
```

The key insight: **no new data is created** -- pixel shuffle is a pure rearrangement.
The input and output have exactly the same number of elements.

---

## 2. Where It Appears in the Mochi Decoder

The Mochi VAE decoder is an asymmetric convolutional decoder with **19 ResBlocks**
(no attention layers) organized into 5 stages:

```
Input: [1, 12, 8, 60, 106] (latents)
  |
  conv_in (12 -> 768)             -- 1x1x1 conv, boundary layer
  |
  block_in: 3x ResBlock(768)
  |
  up_block_0: 6x ResBlock(768) + PIXEL SHUFFLE (768 -> 512, T*3, H*2, W*2)
  |
  up_block_1: 4x ResBlock(512) + PIXEL SHUFFLE (512 -> 256, T*2, H*2, W*2)
  |
  up_block_2: 3x ResBlock(256) + PIXEL SHUFFLE (256 -> 128, T*1, H*2, W*2)
  |
  block_out: 3x ResBlock(128)
  |
  proj_out (128 -> 3)             -- linear, boundary layer
  |
Output: [1, 3, 48, 480, 848] (video)
```

Each `MochiUpBlock3D` performs the pixel shuffle in its `forward()` method
(`diffusers/models/autoencoders/autoencoder_kl_mochi.py`, lines 391-403):

```python
# After linear projection expands channels: C -> C_out * st * sh * sw
hidden_states = hidden_states.view(B, C_out, st, sh, sw, T, H, W)     # 5D -> 8D
hidden_states = hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)         # interleave
hidden_states = hidden_states.view(B, C_out, T*st, H*sh, W*sw)         # 8D -> 5D
```

The permutation `(0, 1, 5, 2, 6, 3, 7, 4)` maps dimensions as:

```
Before: [B, C_out, st, sh, sw, T,  H,  W ]
         0   1      2   3   4   5   6   7

After:  [B, C_out, T,  st, H,  sh, W,  sw]
         0   1      5   2   6   3   7   4
```

This places each expansion factor right after its spatial dimension so the final
reshape can merge `(T, st) -> T*st`, `(H, sh) -> H*sh`, `(W, sw) -> W*sw`.

### Shape trace through all three pixel shuffles

| Block | Input -> Linear Output | 8D Reshape | 8D Permute | 5D Reshape |
|-------|----------------------|------------|------------|------------|
| up_block_0 | `[1,768,8,60,106]` -> `[1,6144,8,60,106]` | `[1,512,3,2,2,8,60,106]` | `[1,512,8,3,60,2,106,`**2**`]` | `[1,512,24,120,212]` |
| up_block_1 | `[1,512,24,120,212]` -> `[1,2048,24,120,212]` | `[1,256,2,2,2,24,120,212]` | `[1,256,24,2,120,2,212,`**2**`]` | `[1,256,48,240,424]` |
| up_block_2 | `[1,256,48,240,424]` -> `[1,512,48,240,424]` | `[1,128,1,2,2,48,240,424]` | `[1,128,48,1,240,2,424,`**2**`]` | `[1,128,48,480,848]` |

Notice: **the last dimension is always 2** (`sw`, the spatial width expansion factor).
This is the root cause of the problem.

---

## 3. The Problem: Tile Alignment on Small Trailing Dimensions

### How tt-mlir tiles tensors

Tenstorrent hardware operates on **32x32 tiles**. When a tensor is placed in DRAM,
the tt-mlir compiler pads its last two dimensions to multiples of 32 using
`getTilePaddedShape()` (from `lib/Dialect/TTNN/Utils/Utils.cpp:213`):

```cpp
// Only the last two dimensions are padded. All higher dims are untouched.
tiledShape[rank - 2] = alignUp(shape[rank - 2], 32);   // TILE_HEIGHT
tiledShape[rank - 1] = alignUp(shape[rank - 1], 32);   // TILE_WIDTH
```

For tensors where the last two dimensions are already close to multiples of 32, the
overhead is minimal:

```
[1, 512, 24, 120, 212]  ->  [1, 512, 24, 128, 224]
                                        120->128  212->224
Overhead: 1.067x * 1.057x = 1.13x  (13% waste -- acceptable)
```

### What goes wrong with the pixel shuffle intermediate

After the permute, the **last dimension becomes `sw = 2`**:

```
[1, 512, 8, 3, 60, 2, 106, 2]  ->  [1, 512, 8, 3, 60, 2, 128, 32]
                                                         106->128  2->32
Overhead: 1.208x * 16x = 19.32x  (1,832% waste!)
```

The tile padding rounds dimension 2 up to 32, wasting **15 out of every 16 elements**
(93.75%) in the innermost dimension. This is not a bug in the allocator -- it is a
fundamental consequence of putting a tiny dimension in the tile-padded position.

### Dummy example: why dim=2 in tile position is catastrophic

Consider a simple `[4, 2]` tensor in bf16 (8 elements, 16 bytes):

```
Logical:   [4, 2]  =  8 elements  =  16 bytes
Tiled:     [32, 32] = 1,024 elements = 2,048 bytes
Overhead:  128x
```

```
  Logical (4x2):          Tiled (32x32):
  ┌───┬───┐               ┌───┬───┬───┬─── ... ───┐
  │ a │ b │               │ a │ b │ 0 │ 0  ... │ 0 │  <- 30 zeros
  │ c │ d │               │ c │ d │ 0 │ 0  ... │ 0 │
  │ e │ f │               │ e │ f │ 0 │ 0  ... │ 0 │
  │ g │ h │               │ g │ h │ 0 │ 0  ... │ 0 │
  └───┴───┘               │ 0 │ 0 │ 0 │ 0  ... │ 0 │  <- 28 rows of zeros
                           │   ...                   │
                           │ 0 │ 0 │ 0 │ 0  ... │ 0 │
                           └───┴───┴───┴─── ... ───┘
                            4 cols of data, 32 cols allocated
                            4 rows of data, 32 rows allocated
```

Every tile is 32x32 but only 4x2 of it contains real data. The rest is padding zeros
that still consume DRAM.

Now scale this to the Mochi decoder pixel shuffle. The 8D permute output has shape
`[1, 512, 8, 3, 60, 2, 106, 2]`:

- The **last dim = 2** pads to **32** (16x waste)
- The **second-to-last dim = 106** pads to **128** (1.21x waste)
- All 6 outer dims (1 * 512 * 8 * 3 * 60 * 2 = 1,474,560) multiply the waste

```
Logical:  312,606,720 elements  *  2 bytes  =    596 MB
Tiled:  6,039,797,760 elements  *  2 bytes  = 11,520 MB  (11.25 GB)
                                                ─────────
                                                19.32x overhead
```

A tensor that should be 596 MB occupies **11.25 GB** of DRAM on a single 32 GB device.

---

## 4. Verified Impact: tt-swiss Memory Report

The tt-swiss memory report (`/root/.ttmem/reports/decoder-sharded/`) confirms this
analysis. The report tracks per-operation DRAM allocation across 8 DRAM banks on a
single Blackhole device (32 GB total).

### Memory trajectory around the first pixel shuffle (up_block_0)

| Op Index | Operation | Shape | Alloc/Bank | Total Alloc |
|----------|-----------|-------|------------|-------------|
| 1292 | Linear projection | 50880x6144 | 482 MB | 3,858 MB |
| 1295 | Permute (5D->5D) | 1x6144x8x60x106 | 636 MB | 5,090 MB |
| 1297 | Reshape (5D->8D) | 1x512x3x2x2x8x60x106 | 732 MB | 5,858 MB |
| **1298** | **Permute (8D->8D)** | **1x512x8x3x60x2x106x2** | **732 MB** | **5,858 MB** |
| **1299** | **Reshape (8D->5D)** | **1x512x24x120x212** | **2,172 MB** | **17,378 MB** |
| 1308 | (tensors settled) | ... | 2,251 MB | 18,010 MB |

The 11,520 MB allocation materializes at op 1299 (the reshape that consumes the 8D
tensor). After the pixel shuffle completes, live tensor data is only ~98 MB, but the
allocator has consumed 18,010 MB -- **~14 GB of permanent fragmentation** from the
spike.

### What the report covers

The report contains **1,813 operations** covering only **59% of the decoder**:

| Stage | ResBlocks | Status |
|-------|-----------|--------|
| conv_in + block_in | 3/3 | Complete |
| up_block_0 (6 ResBlocks + pixel shuffle) | 6/6 + shuffle | Complete |
| up_block_1 (4 ResBlocks + pixel shuffle) | **2/4** | **Partial -- truncated** |
| up_block_2 (3 ResBlocks + pixel shuffle) | 0/3 | Missing |
| block_out | 0/3 | Missing |

The report **truncates mid-way through up_block_1** (op 1812, during reflect-padding
for the 3rd ResBlock). The model hits OOM before ever reaching the second pixel shuffle.

### Peak memory at truncation

```
Peak: op 1762 (ttnn.sum in GroupNorm)
  Allocated:  3,372 MB/bank  =  26,979 MB total  (82.7% of 32 GB)
  Free:         703 MB/bank  =   5,624 MB total
  Contiguous:   336 MB/bank  =   2,688 MB contiguous free

Of the 26,979 MB allocated, ~14,000 MB is fragmentation from the pixel shuffle.
Only ~13,000 MB is actual live data.
```

---

## 5. Why the Decoder Cannot Fit in 128 GB (4x32 GB)

### The sharding helps for ResBlocks, not for pixel shuffle

The Megatron-style channel tensor-parallel sharding divides ResBlock activations across
4 devices. This works well: the largest unsharded activation (4.66 GB in block_out)
becomes ~1.17 GB per device.

However, **the pixel shuffle (unpatchify) runs on replicated data**. The linear
projection and subsequent reshape/permute/reshape all execute with the full tensor on
every device. This means each device independently creates the tile-padded monster.

### Per-device memory for each pixel shuffle (replicated, tiled)

| Pixel Shuffle | Logical Size | Tiled Size (per device) | Fits in 32 GB? |
|---------------|-------------|------------------------|----------------|
| up_block_0 | 596 MB | **11.25 GB** | Yes, but fragments 14 GB |
| up_block_1 | 2.33 GB | **39.4 GB** | **NO (exceeds 32 GB)** |
| up_block_2 | 4.66 GB | **78.7 GB** | **NO (exceeds 32 GB)** |

**The second pixel shuffle alone requires 39.4 GB on each device.** Even with no
fragmentation from the first pixel shuffle, this tensor physically cannot fit in a
single 32 GB device. The third pixel shuffle (78.7 GB) is even worse.

### Why sharding the pixel shuffle across 4 devices would also be challenging

If the pixel shuffle intermediates were channel-sharded (4-way):

| Pixel Shuffle | Tiled/device (sharded 4-way) | Fits? |
|---------------|------------------------------|-------|
| up_block_0 | 2.81 GB | Yes |
| up_block_1 | 9.85 GB | Tight (30% of DRAM) |
| up_block_2 | 19.7 GB | Very tight (61% of DRAM) |

This helps but doesn't eliminate the 16x fundamental waste, and up_block_2 at 19.7 GB
would leave only 12.3 GB for all weights, other activations, and compiler overhead.

### The math is clear

Without fixing the tiling issue, the Mochi decoder **cannot run** on 4x32 GB hardware.
The pixel shuffle creates intermediates that are 16-19x larger than their logical size
due to tile padding of `dim=2`, and no amount of sharding can compensate for a 16x
memory multiplier on the largest tensors in the model.

---

## 6. Options for Fixing It

### Option A: Rewrite the Pixel Shuffle Permute Order (Model-Level, Immediate)

**Approach**: Change the permutation so the last dimension is always the spatial width
(W, which is large: 106/212/424) rather than the expansion factor (sw=2).

The current permutation:
```python
# Current: puts sw=2 as last dim
x = x.view(B, C, st, sh, sw, T, H, W)          # [B,C,st,sh,sw,T,H,W]
x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)          # [B,C,T,st,H,sh,W,sw]
#                  last dim = sw = 2 ─────────────────────────────────┘
x = x.view(B, C, T*st, H*sh, W*sw)
```

A possible rewrite:
```python
# Option A1: two-step permute, keep W in last position
x = x.view(B, C, st, sh, sw, T, H, W)          # [B,C,st,sh,sw,T,H,W]
x = x.permute(0, 1, 5, 2, 6, 3, 4, 7)          # [B,C,T,st,H,sh,sw,W]
#                  last dim = W >= 106 ───────────────────────────────┘
x = x.reshape(B, C, T*st, H*sh, sw*W)           # merge sw*W (not W*sw)
```

Note: `sw*W` and `W*sw` produce the same dimension size, but the element ordering
differs. This would produce a different (incorrect) output because the interleaving
pattern changes. **The output pixels would be in the wrong order.**

To get the correct output, we'd need to perform an additional permute or use a
different decomposition of the pixel shuffle that doesn't put small factors last.

```python
# Option A2: do height/width expansion separately
# Step 1: temporal expansion (no small trailing dims if W >= 32)
x = x.view(B, C * sh * sw, st, T, H, W)
x = x.permute(0, 1, 3, 2, 4, 5)                # [B,C*sh*sw,T,st,H,W]
x = x.reshape(B, C * sh * sw, T * st, H, W)

# Step 2: spatial expansion
x = x.view(B, C, sh, sw, T * st, H, W)
x = x.permute(0, 1, 4, 5, 2, 6, 3)             # [B,C,T*st,H,sh,W,sw]
#                          last dim = sw = 2 ──────────────────────────┘  STILL 2!
```

This still leaves sw=2 as the last dim. The fundamental problem: any permutation that
interleaves a factor of 2 into spatial dims will, at some point, need it as a trailing
dimension (unless the reshape can absorb it differently).

```python
# Option A3: use repeat_interleave (no 8D tensor at all)
# Linear projects C -> C_out (no expansion factors in channels)
x = self.proj(x)     # [B, C_out, T, H, W] -- standard projection
# Then expand spatially using repeat_interleave
x = x.repeat_interleave(st, dim=2)              # T -> T*st
x = x.repeat_interleave(sh, dim=3)              # H -> H*sh
x = x.repeat_interleave(sw, dim=4)              # W -> W*sw
```

**However**, `repeat_interleave` actually **duplicates** data rather than rearranging
it. Pixel shuffle rearranges unique channel values into spatial positions, while
`repeat_interleave` would copy the same spatial value multiple times. These produce
different results -- this is NOT a valid replacement unless the linear projection is
also changed to output `C_out` channels instead of `C_out * st * sh * sw`.

**Verdict**: A pure model-level permute rewrite is **difficult** because the 8D
interleaving is mathematically required to correctly scatter channel values into
spatiotemporal positions. Any decomposition that produces the correct output will at
some point have a small trailing dimension. The most promising model-level approach
would be to restructure the linear projection to avoid needing the 8D intermediate
entirely (e.g., separate linear projections per spatial factor), but this changes the
model architecture.

**Difficulty**: High -- requires careful mathematical equivalence verification.
**Impact**: Eliminates the problem entirely if a valid decomposition is found.

---

### Option B: Force ROW_MAJOR Layout for Tile-Hostile Intermediates (Compiler Fix)

**Approach**: Modify the `TTNNLayout` pass to detect when tiling would cause excessive
overhead and keep those tensors in ROW_MAJOR (untiled) layout.

**Where**: `lib/Dialect/TTNN/Transforms/TTNNLayout.cpp`, function `shouldTilizeResult()`

The pass already has precedent -- Conv3d results are forced to ROW_MAJOR:
```cpp
// Lines 288-291: Conv3d outputs are not tilized
if (isa<Conv3dOp>(op)) {
    return false;  // ROW_MAJOR
}
```

Add a similar rule for tensors with extreme tile overhead:
```cpp
// Proposed: skip tilization when padding overhead exceeds threshold
auto shape = resultType.getShape();
int rank = shape.size();
if (rank >= 2) {
    int64_t lastDim = shape[rank - 1];
    int64_t padded = alignUp(lastDim, TILE_WIDTH);
    // If padding would waste >50% of the last dimension, stay ROW_MAJOR
    if (padded > 2 * lastDim) {
        return false;
    }
}
```

For the pixel shuffle intermediate (`lastDim=2`, `padded=32`): `32 > 2*2 = 4` -- true,
stays ROW_MAJOR. The tensor remains at 596 MB instead of 11,520 MB.

For a normal tensor (`lastDim=212`, `padded=224`): `224 > 2*212 = 424` -- false,
gets tilized as usual.

**Difficulty**: Low-medium -- small, localized compiler change with clear precedent.
**Impact**: Fixes all three pixel shuffles. The 8D intermediate exists but at logical
size (~596 MB, ~2.33 GB, ~4.66 GB) rather than 16x inflated.
**Risk**: Downstream ops may expect tiled input. The reshape that follows the permute
should handle ROW_MAJOR input, but needs verification. A `to_layout` (ROW_MAJOR->TILE)
conversion would be needed before the next tiled op.

---

### Option C: Fuse Reshape-Permute-Reshape in the Compiler (Compiler Pass)

**Approach**: Add a pattern that recognizes the pixel shuffle sequence
(reshape 5D->8D, permute, reshape 8D->5D) and replaces it with a single fused op that
writes directly to the 5D output without materializing the 8D intermediate.

**Where**: `lib/Dialect/TTIR/IR/TTIRTMFusionPatterns.cpp`

The existing `PermuteReshapePermuteFusionPattern` fuses `Permute->Reshape->Permute`.
A new `ReshapePermuteReshapeFusionPattern` would fuse `Reshape->Permute->Reshape` when
it matches the depth-to-space-time pattern.

The fused operation would:
1. Read from the 5D input tensor (post-linear-projection)
2. Compute the output coordinates using the interleaving formula
3. Write to the 5D output tensor
4. Never allocate the 8D intermediate

**Difficulty**: Medium-high -- requires implementing the fusion pattern matching and a
custom TTNN lowering for the fused op.
**Impact**: Eliminates the 8D intermediate entirely. Optimal solution long-term.
**Risk**: The fused op needs an efficient implementation in tt-metal/TTNN.

---

### Option D: Shard the Pixel Shuffle Path (Model-Level)

**Approach**: Extend the sharding annotations to include the unpatchify linear
projection and pixel shuffle, so the intermediates are channel-sharded across 4 devices
rather than replicated.

Currently in `decoder_sharded.py`:
```python
# up_block.proj (unpatchify linear) is left unsharded
```

If we shard the linear projection output along the channel dimension, the 8D
intermediate would also be channel-sharded, reducing per-device memory by 4x:

| Pixel Shuffle | Tiled (replicated) | Tiled (4-way sharded) |
|---------------|-------------------|-----------------------|
| up_block_0 | 11.25 GB | 2.81 GB |
| up_block_1 | 39.4 GB | 9.85 GB |
| up_block_2 | 78.7 GB | 19.7 GB |

**Difficulty**: Medium -- requires sharding annotations for the linear projection and
verification that the permute/reshape work correctly with sharded channel dim.
**Impact**: Reduces per-device pressure by 4x, but up_block_2 at 19.7 GB is still
very tight. Does NOT fix the fundamental 16x waste.
**Risk**: The reshape and permute ops may not preserve sharding correctly through the
8D intermediate. SPMD/Shardy support for 8D tensors needs verification.

---

### Option E: Insert Memory Compaction After Pixel Shuffle (Runtime)

**Approach**: Insert `ttnn.reallocate()` (which calls `ttnn::move()`) after the pixel
shuffle tensor is freed to compact surviving allocations and recover fragmented memory.

**Where**: Could be inserted by a compiler pass in `TTNNMemoryManagement.cpp` or as a
post-processing step.

**Difficulty**: Low.
**Impact**: Only recovers fragmentation from up_block_0's pixel shuffle. Does NOT help
with up_block_1 and up_block_2 which are simply too large to fit even without
fragmentation. This is a complementary optimization, not a standalone fix.

---

### Recommended Strategy

The options are not mutually exclusive. The recommended approach combines them:

| Priority | Option | Why |
|----------|--------|-----|
| **1st** | **B (ROW_MAJOR threshold)** | Lowest effort, fixes all three pixel shuffles, clear precedent in the codebase. This alone makes the decoder runnable. |
| **2nd** | **D (Shard pixel shuffle)** | Further reduces per-device memory. Combined with B, up_block_2 goes from 4.66 GB/device (replicated, ROW_MAJOR) to 1.17 GB/device. |
| **3rd** | **E (Memory compaction)** | Recovers fragmentation from any remaining large transient allocations. |
| **Long-term** | **C (Fuse ops)** | Eliminates the 8D intermediate entirely for optimal performance. |

With Option B alone, the pixel shuffle intermediates would be:

| Pixel Shuffle | Tiled (current) | ROW_MAJOR | Savings |
|---------------|-----------------|-----------|---------|
| up_block_0 | 11.25 GB | 596 MB | **18.9x** |
| up_block_1 | 39.4 GB | 2.33 GB | **16.9x** |
| up_block_2 | 78.7 GB | 4.66 GB | **16.9x** |

All three fit comfortably in a single 32 GB device, leaving ample room for weights
(~700 MB replicated) and activations.

---

## Appendix: Detailed Tile Padding Calculations

### Tile padding rule

```
getTilePaddedShape(shape):
    shape[-2] = ceil(shape[-2] / 32) * 32    # TILE_HEIGHT = 32
    shape[-1] = ceil(shape[-1] / 32) * 32    # TILE_WIDTH  = 32
    # All other dimensions unchanged
```

Source: `tt-mlir/lib/Dialect/TTNN/Utils/Utils.cpp:213-225`

### All pixel shuffle intermediates

**up_block_0 post-permute**: `[1, 512, 8, 3, 60, 2, 106, 2]`
```
dim[-2]: 106 -> ceil(106/32)*32 = 128    (1.21x)
dim[-1]:   2 -> ceil(2/32)*32   =  32    (16.0x)

Logical:  1*512*8*3*60*2*106*2      = 312,606,720 elements =    596 MB (bf16)
Padded:   1*512*8*3*60*2*128*32     = 6,039,797,760 elements = 11,520 MB (bf16)
Overhead: 19.32x
```

**up_block_1 post-permute**: `[1, 256, 24, 2, 120, 2, 212, 2]`
```
dim[-2]: 212 -> ceil(212/32)*32 = 224    (1.06x)
dim[-1]:   2 -> ceil(2/32)*32   =  32    (16.0x)

Logical:  1*256*24*2*120*2*212*2    = 1,250,426,880 elements =  2,330 MB (bf16)
Padded:   1*256*24*2*120*2*224*32   = 21,139,292,160 elements = 40,240 MB (bf16)
Overhead: 16.91x
```

**up_block_2 post-permute**: `[1, 128, 48, 1, 240, 2, 424, 2]`
```
dim[-2]: 424 -> ceil(424/32)*32 = 448    (1.06x)
dim[-1]:   2 -> ceil(2/32)*32   =  32    (16.0x)

Logical:  1*128*48*1*240*2*424*2    = 2,500,853,760 elements =  4,660 MB (bf16)
Padded:   1*128*48*1*240*2*448*32   = 42,278,584,320 elements = 80,480 MB (bf16)
Overhead: 16.91x
```

### Comparison: normal 5D tensor (post-pixel-shuffle output)

**up_block_0 output**: `[1, 512, 24, 120, 212]`
```
dim[-2]: 120 -> 128    (1.07x)
dim[-1]: 212 -> 224    (1.06x)

Logical:  312,606,720 elements  =  596 MB
Padded:   352,321,536 elements  =  672 MB
Overhead: 1.13x  (13% -- perfectly acceptable)
```

### Verification against tt-swiss report

The tt-swiss memory report for op 1298 reports a padded output of **11,520 MB**.
Our calculation: `6,039,797,760 elements * 2 bytes / (1024*1024) = 11,520.0 MB`.
**Exact match.** The analysis is confirmed.
