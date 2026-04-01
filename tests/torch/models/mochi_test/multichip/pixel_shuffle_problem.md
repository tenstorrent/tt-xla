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
single Blackhole device (32 GB total, 4,075 MB per bank).

### 4.1 The full pixel shuffle pipeline: op-by-op

The pixel shuffle for up_block_0 spans ops 1285-1320. Below is **every operation**,
showing both physical DRAM allocation (what the allocator reserves) and logical tensor
memory (padded vs unpadded). This lets us pinpoint exactly which op causes the explosion.

#### Phase 1: Preparation (ops 1285-1291)

The ResBlock output feeds into a channel-last permute and linear projection:

| Op | Type | Source | Output Shape | Padded (MB) | Alloc/Bank | Delta/Bank | Live Tensors |
|----|------|--------|-------------|-------------|------------|------------|--------------|
| 1285 | ttnn.add | aten\_\_add | 1x768x8x60x106 | 96 | 474.0 | +0.0 | 2 |
| 1286 | ttnn.to\_layout | layout convert | 1x768x8x60x106 | 96 | 461.3 | -12.8 | 2 |
| 1287 | ttnn.permute | aten\_\_permute | 1x8x60x106x768 | 90 | 473.3 | +12.0 | 2 |
| 1288 | ttnn.to\_layout | layout convert | 1x8x60x106x768 | 90 | 471.8 | -1.5 | 3 |
| 1289 | ttnn.reshape | reshape for matmul | 50880x768 | 75 | 483.0 | +11.3 | 3 |
| 1290 | ttnn.to\_layout | weight prep | 6144x768 | 9 | 481.1 | -1.9 | 4 |
| 1291 | ttnn.to\_layout | bias prep | 6144 | 0.4 | 482.2 | +1.1 | 5 |

Nothing unusual here. Memory fluctuates modestly as tensors are reshaped and prepared
for the linear projection. Overhead is minimal (all dims are large enough for
reasonable tile alignment).

#### Phase 2: Linear projection (op 1292)

| Op | Type | Source | Output Shape | Padded (MB) | Alloc/Bank | Delta/Bank | Unpadded Total | Padded Total |
|----|------|--------|-------------|-------------|------------|------------|----------------|--------------|
| **1292** | **ttnn.linear** | **aten\_\_add** | **50880x6144** | **596** | **482.2** | **+0.05** | **1,267 MB** | **1,536 MB** |

The linear projection expands channels from 768 to 6144 (= 512 * 3 * 2 * 2). The
output is 596 MB with **zero tiling overhead** -- both dimensions (50880, 6144) are
large and tile-aligned. Memory barely changes because old intermediates are freed.

**Overhead ratio: 1,536 / 1,267 = 1.21x (21%) -- perfectly normal.**

#### Phase 3: Reshape to channels-first 5D (ops 1293-1296)

| Op | Type | Source | Output Shape | Padded (MB) | Alloc/Bank | Delta/Bank | Unpadded Total | Padded Total | Overhead |
|----|------|--------|-------------|-------------|------------|------------|----------------|--------------|----------|
| 1293 | ttnn.reshape | aten\_\_add | 1x8x60x106x6144 | 720 | 546.3 | +64.0 | 1,267 | 1,536 | 1.21x |
| 1294 | ttnn.to\_layout | layout convert | 1x8x60x106x6144 | 720 | 546.3 | +0.0 | 1,267 | 1,536 | 1.21x |
| 1295 | ttnn.permute | aten\_\_permute | 1x6144x8x60x106 | 768 | 636.3 | +90.0 | 1,267 | 1,632 | 1.29x |
| 1296 | ttnn.to\_layout | layout convert | 1x6144x8x60x106 | 768 | 636.3 | +0.0 | 1,267 | 1,632 | 1.29x |

Reshaping from 2D matmul output back to 5D, then permuting to channels-first layout
`[1, 6144, 8, 60, 106]`. The 5D tensor is 768 MB padded (vs 596 MB logical = 1.29x
overhead from 60->64 and 106->128 padding). **Still reasonable.**

#### Phase 4: THE EXPLOSION -- reshape to 8D + permute (ops 1297-1299)

| Op | Type | Source | Output Shape | Padded Output (MB) | Alloc/Bank | Delta/Bank | Unpadded Total | Padded Total | **Overhead** |
|----|------|--------|-------------|---------------------|------------|------------|----------------|--------------|----------|
| 1297 | ttnn.reshape | aten\_\_view | 1x512x3x2x2x8x60x106 | 768 | 732.3 | +96.0 | 1,863 | **2,352** | **1.26x** |
| **1298** | **ttnn.permute** | **aten\_\_permute** | **1x512x8x3x60x2x106x2** | **11,520** | **732.3** | **+0.0** | **1,863** | **13,104** | **7.03x** |
| **1299** | **ttnn.reshape** | **aten\_\_view** | **1x512x24x120x212** | **672** | **2,172.3** | **+1,440.0** | **2,460** | **13,776** | **5.60x** |

**This is the critical sequence. Let's walk through each op:**

**Op 1297** -- `reshape [1,6144,8,60,106] -> [1,512,3,2,2,8,60,106]`:
Factoring 6144 = 512 * 3 * 2 * 2 into separate dimensions. This is a **metadata-only**
reshape -- same physical data, same tiling. Last two dims are still (60, 106) which
tile to (64, 128). Padded output is 768 MB. Overhead is only 1.26x. **Nothing wrong
here.**

**Op 1298** -- `permute [1,512,3,2,2,8,60,106] -> [1,512,8,3,60,2,106,2]`:
This is the interleaving permute. It moves the last dim from 106 (large) to 2 (tiny).
The padded output is **11,520 MB** -- a 19.32x blowup from the 596 MB of logical data.

> **Critical detail**: At op 1298, `alloc/bank` does NOT change (stays at 732.3 MB).
> The permute is initially represented as a **lazy view** -- the runtime has not yet
> allocated the padded physical memory. But the aggregate `padded_total` jumps from
> 2,352 to 13,104 MB (+10,752 MB), showing the future allocation requirement.

**Op 1299** -- `reshape [1,512,8,3,60,2,106,2] -> [1,512,24,120,212]`:
This reshape consumes the 8D tensor and produces a well-shaped 5D output (672 MB).
**This is where the 8D tensor is physically materialized.** The allocator must allocate
the full 11,520 MB to read the permuted data and write the 5D output:

```
alloc/bank: 732.3 -> 2,172.3 MB  (+1,440.0 MB/bank)
Total:    5,858   -> 17,378  MB  (+11,520 MB across 8 banks)
                                   ^^^^^^^ exactly the padded 8D tensor size
```

#### Phase 5: Aftermath -- memory never recovers (ops 1300-1320)

| Op | Type | Source | Output Shape | Alloc/Bank | Unpadded Total | Padded Total | Overhead |
|----|------|--------|-------------|------------|----------------|--------------|----------|
| 1300 | ttcore.load\_cached | const eval | 1x1x1x1 | 2,251.0 | 2,460 | 2,256 | 0.92x |
| 1301 | ttnn.full | -- | -- | 2,251.0 | -- | -- | -- |
| 1302 | ttnn.reshape | aten\_\_sub\_tm1 | 1x1x1x1 | 2,251.0 | 671 | **2,256** | **3.36x** |
| ... | (GroupNorm weight/bias prep) | ... | ... | ~2,251 | -- | -- | -- |
| 1316 | ttnn.to\_layout | layout convert | 1x512x24x120x212 | 2,251.1 | 1,440 | 1,440 | 1.00x |
| 1317 | ttnn.permute | aten\_\_permute | 1x24x512x120x212 | 2,335.1 | 1,440 | 1,536 | 1.07x |
| 1318 | ttnn.reshape | aten\_\_view | 24x512x120x212 | 2,335.1 | 1,440 | 1,536 | 1.07x |
| 1319 | ttnn.slice\_static | xla\_\_select | 8x512x120x212 | 2,335.1 | -- | -- | -- |
| 1320 | ttnn.typecast | xla\_\_cast | 8x512x120x212 | 2,363.1 | -- | -- | -- |

**Key observation at op 1302**: The padded total drops from 13,776 to 2,256 MB --
the 8D tensor is logically dead. **But `alloc/bank` stays at 2,251 MB.** The physical
DRAM is never reclaimed. The free-list allocator has the memory marked as free, but
surviving small allocations (constants loaded at op 1300) split the freed region into
non-contiguous fragments.

```
Memory state after pixel shuffle settles (op ~1308):
  ┌────────────────────────────────────────────────────────────┐
  │                    DRAM Bank (4,075 MB)                     │
  ├──────────┬──────────────────────────┬──────────┬───────────┤
  │ weights  │   freed (fragmented)     │ const    │   free    │
  │ ~453 MB  │ hole from 11,520 MB      │ ~79 MB   │ 1,824 MB  │
  │          │ alloc (never compacted)  │          │           │
  ├──────────┴──────────────────────────┴──────────┴───────────┤
  │ alloc/bank = 2,251 MB                free/bank = 1,824 MB  │
  │ But live tensor data ≈ 98 MB         Contiguous ≈ 1,824 MB │
  └────────────────────────────────────────────────────────────┘
```

The 11,520 MB allocation pushed the high-water mark up. Even though the memory was
freed, a small constant allocated at op 1300 sits at a higher address, preventing the
allocator from compacting the free space back down to the pre-pixel-shuffle level.

### 4.2 The full memory trajectory visualization

```
alloc/bank (MB)
  │
  │                                                           Peak: 3,372
  │                                                          ╱
3000├──────────────────────────────────────────────────────── ╱ ──
  │                                                     ╱  ╱
  │                                          up_block_1 ╱  ╱
2500├──────────────────────────────────────── ╱ ────────╱──╱───
  │                              ╱         ╱        ╱
  │                        ╱   ╱ conv3d  ╱ conv3d ╱
2000├────────────────── ╱──╱───╱─────────╱───────╱─────────
  │              ╱   ╱  ╱
  │        ╱╱╱╱╱  ╱  ╱  ╱  up_block_1 ResBlocks
  │     ╱╱╱╱    ╱  ╱  ╱
  │  ╱╱╱       ╱  ╱  ╱                                 ◄── OOM at op 1812
1500├────────╱──╱──╱──────────────────────────────────────
  │       ╱  ╱  ╱
  │      ╱  ╱  ╱
  │     ╱  ╱  ╱
1000├───╱──╱──╱───────────────────────────────────────────
  │  ╱  ╱  ╱
  │ ╱  ╱  ╱
  │╱  ╱  ╱
 500├─╱──╱────────────────────────────────────────────────
  │╱╱                ▲
  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓█
  │   block_in +     █ PIXEL SHUFFLE
  │   up_block_0     █ +1,440 MB/bank
  │   9 ResBlocks    █ in ONE op
  0├─────────────────────────────────────────────────────→ op index
   0   200  400  600  800 1000 1200 1400 1600 1800
                               ▲1298              ▲1812
                          pixel shuffle         report ends
```

The memory trajectory has two distinct phases:
1. **Ops 0-1297** (block\_in + up\_block\_0 ResBlocks): Slow, steady growth from 0 to
   732 MB/bank. Each ResBlock pair adds ~47 MB/bank of permanent weight storage.
2. **Op 1298-1299** (pixel shuffle): **Instantaneous cliff** from 732 to 2,172 MB/bank
   (+1,440 MB/bank = +11,520 MB). This single operation consumes more DRAM than all
   preceding 1,297 operations combined.
3. **Ops 1300-1812** (up\_block\_1 ResBlocks): Continued growth from 2,172 to 3,372
   MB/bank until OOM.

### 4.3 What the report covers (and doesn't)

The report contains **1,813 operations** covering only **59% of the decoder**:

| Stage | ResBlocks | Status |
|-------|-----------|--------|
| conv\_in + block\_in | 3/3 | Complete |
| up\_block\_0 (6 ResBlocks + pixel shuffle) | 6/6 + shuffle | Complete |
| up\_block\_1 (4 ResBlocks + pixel shuffle) | **2/4** | **Partial -- truncated** |
| up\_block\_2 (3 ResBlocks + pixel shuffle) | 0/3 | Missing |
| block\_out | 0/3 | Missing |

The report **truncates mid-way through up\_block\_1** (op 1812, during reflect-padding
for the 3rd ResBlock). The model hits OOM before ever reaching the second pixel shuffle.

### 4.4 Peak memory at truncation

```
Peak: op 1762 (ttnn.sum in GroupNorm)
  Allocated:  3,372 MB/bank  =  26,979 MB total  (82.7% of 32 GB)
  Free:         703 MB/bank  =   5,624 MB total
  Contiguous:   336 MB/bank  =   2,688 MB contiguous free

Of the 26,979 MB allocated, ~14,000 MB is fragmentation from the pixel shuffle.
Only ~13,000 MB is actual live data.
```

### 4.5 Where exactly is the problem?

To summarize the per-op analysis:

| Op | What Happens | Padded Output | Is It a Problem? |
|----|-------------|---------------|------------------|
| 1292 | Linear 768->6144 | 596 MB | **NO** -- zero overhead, both dims tile-aligned |
| 1293 | Reshape to 5D (channels-last) | 720 MB | **NO** -- 1.21x overhead, last dim = 6144 |
| 1295 | Permute to channels-first | 768 MB | **NO** -- 1.29x overhead, last dim = 106 |
| 1297 | Reshape 5D->8D (factor channels) | 768 MB | **NO** -- same physical layout, last dim = 106 |
| **1298** | **Permute 8D (interleave)** | **11,520 MB** | **YES -- last dim becomes 2, 19.32x overhead** |
| 1299 | Reshape 8D->5D (merge spatial) | 672 MB | The output is fine (1.13x), but it **materializes op 1298's allocation** |

**The problem is isolated to exactly one operation: the 8D permute at op 1298.** Every
other op in the pixel shuffle pipeline has reasonable tiling overhead (<1.3x). The
permute is the sole cause because it moves `sw=2` into the last dimension position
where tile padding inflates it 16x.

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

## 6. Solutions

### Solution 1: Staged Pixel Shuffle Decomposition (Model-Level Patch)

**Status: Implemented and verified. Produces bit-identical output.**

Instead of merging all 3 dimension pairs in a single reshape (which requires the
catastrophic 8D permute), merge **one pair at a time**, using intermediate permutes
to keep large dimensions in the tile-padded positions at every step.

#### The rewrite

```python
# ORIGINAL: 1 permute + 1 reshape (last dim = sw = 2, causes 16-19x overhead)
x = x.view(B, C, st, sh, sw, T, H, W)              # [B, C, st, sh, sw, T, H, W]
x = x.permute(0, 1, 5, 2, 6, 3, 7, 4)              # [B, C, T, st, H, sh, W, sw]
x = x.view(B, C, T*st, H*sh, W*sw)

# STAGED: 3 permutes + 2 reshapes (last 2 dims always large, max 1.33x overhead)
x = x.view(B, C, st, sh, sw, T, H, W)              # [B, C, st, sh, sw, T, H, W]
x = x.permute(0, 1, 5, 2, 3, 4, 6, 7)              # [B, C, T, st, sh, sw, H, W]     last2=(H,W)
x = x.reshape(B, C, T*st, sh, sw, H, W)             # merge (T,st)->T*st               last2=(H,W)
x = x.permute(0, 5, 3, 6, 4, 1, 2)                  # [B, H, sh, W, sw, C, T*st]      last2=(C,T*st)
x = x.reshape(B, H*sh, W*sw, C, T*st)               # merge (H,sh) and (W,sw)          last2=(C,T*st)
x = x.permute(0, 3, 4, 1, 2)                        # [B, C, T*st, H*sh, W*sw]         last2=(H*sh,W*sw)
```

#### Why it works

Each merge handles one or two pairs while the other large dims occupy the tile
positions:

| Step | Shape | Last 2 dims | Tile overhead |
|------|-------|-------------|---------------|
| view 8D | `[B,C,st,sh,sw,T,H,W]` | (H, W) = (60, 106) | 1.29x |
| permute 1 | `[B,C,T,st,sh,sw,H,W]` | (H, W) = (60, 106) | 1.29x |
| reshape 1 | `[B,C,T*st,sh,sw,H,W]` | (H, W) = (60, 106) | 1.29x |
| permute 2 | `[B,H,sh,W,sw,C,T*st]` | (C, T*st) = (512, 24) | 1.33x |
| reshape 2 | `[B,H*sh,W*sw,C,T*st]` | (C, T*st) = (512, 24) | 1.33x |
| permute 3 | `[B,C,T*st,H*sh,W*sw]` | (H*sh, W*sw) = (120, 212) | 1.13x |

The small factors (st=3, sh=2, sw=2) are **never** in the last 2 positions.

#### Peak intermediate memory comparison

| Block | Original (single permute) | Staged (max intermediate) | Reduction |
|-------|--------------------------|---------------------------|-----------|
| up_block_0 | **11,520 MB** (19.3x) | **795 MB** (1.33x) | **14.5x** |
| up_block_1 | **40,320 MB** (16.9x) | **3,180 MB** (1.33x) | **12.7x** |
| up_block_2 | **80,640 MB** (16.9x) | **6,360 MB** (1.33x) | **12.7x** |

All fit comfortably in 32 GB per device.

#### Verification

Bit-identical output verified at full Mochi shapes in bf16 -- see
`test_pixel_shuffle_rewrite.py` for the equivalence test and tile padding analysis.

**Trade-off**: 3 permutes + 2 reshapes instead of 1 + 1. More ops, but the memory
savings (14x fewer MB) far outweigh the extra permute overhead.

---

### Solution 2: ROW_MAJOR Layout for Tile-Hostile Permutes (Compiler Fix)

**Status: Proposed for tt-mlir. See GitHub issue.**

Instead of patching the model, fix this at the compiler level so all models benefit.
Keep the permute and its subsequent reshape in ROW_MAJOR (untiled) layout, and only
convert to TILE on the well-shaped 5D output.

#### The layout cascade

```
reshape [1,6144,8,60,106] -> [1,512,3,2,2,8,60,106]     TILE      768 MB   (1.29x)
permute -> [1,512,8,3,60,2,106,2]                        ROW_MAJOR 596 MB   (no padding)
reshape -> [1,512,24,120,212]                             ROW_MAJOR 596 MB   (no padding)
to_layout(ROW_MAJOR -> TILE)                              TILE      672 MB   (1.13x)
```

#### Required changes in `TTNNLayout.cpp`

Three coordinated modifications:

1. **`shouldTilizeResult` for PermuteOp**: Return false when tile padding would cause
   excessive overhead (e.g., when the last dim < 16, so padding to 32 wastes >50%).
   Precedent: Conv3d already returns false here (line 288-291).

2. **`shouldTilizeResult` for ReshapeOp**: Fix to read the actual input type
   (`reshapeOp.getInput().getType()`) instead of the result type
   (`reshapeOp.getType()`). This appears to be a bug -- the variable is even named
   `inputType` but reads the result type.

3. **Operand forcing for ReshapeOp**: The operand loop unconditionally forces all
   inputs to TILE. For ReshapeOp, it should respect the result layout decision, so a
   ROW_MAJOR reshape doesn't get `to_layout(ROW_MAJOR -> TILE)` inserted before it.

Both `ttnn.permute` and `ttnn.reshape` have verified ROW_MAJOR code paths in tt-metal.

#### Memory savings

| Pixel Shuffle | Current (TILE) | With ROW_MAJOR chain | Fits in 32 GB? |
|---------------|----------------|----------------------|----------------|
| up_block_0 | 11,520 MB | 672 MB | Yes |
| up_block_1 | 40,240 MB | 2,688 MB | Yes |
| up_block_2 | 80,480 MB | 4,928 MB | Yes |

---

### Recommended Strategy

| Priority | Solution | Scope |
|----------|----------|-------|
| **Now** | **Solution 1 (staged decomposition)** | Unblocks Mochi decoder immediately via model-level monkey-patch. No compiler changes. |
| **Next** | **Solution 2 (ROW_MAJOR in tt-mlir)** | Fixes the general problem for any model with tile-hostile permute shapes. Filed as GitHub issue. |

---

## Appendix: Tile Padding Calculations

### Tile padding rule

```
getTilePaddedShape(shape):
    shape[-2] = ceil(shape[-2] / 32) * 32    # TILE_HEIGHT = 32
    shape[-1] = ceil(shape[-1] / 32) * 32    # TILE_WIDTH  = 32
```

Source: `tt-mlir/lib/Dialect/TTNN/Utils/Utils.cpp:213-225`

### Original pixel shuffle intermediates (the problem)

| Block | Post-permute shape | Last 2 dims | Padded to | Logical | Padded | Overhead |
|-------|-------------------|-------------|-----------|---------|--------|----------|
| up_block_0 | `[1,512,8,3,60,2,106,2]` | (106, 2) | (128, 32) | 596 MB | 11,520 MB | 19.32x |
| up_block_1 | `[1,256,24,2,120,2,212,2]` | (212, 2) | (224, 32) | 2,330 MB | 40,240 MB | 16.91x |
| up_block_2 | `[1,128,48,1,240,2,424,2]` | (424, 2) | (448, 32) | 4,660 MB | 80,480 MB | 16.91x |

### Staged decomposition worst step (the fix)

| Block | Worst intermediate shape | Last 2 dims | Padded to | Logical | Padded | Overhead |
|-------|-------------------------|-------------|-----------|---------|--------|----------|
| up_block_0 | `[1,60,2,106,2,512,24]` | (512, 24) | (512, 32) | 596 MB | 795 MB | 1.33x |
| up_block_1 | `[1,120,2,212,2,256,48]` | (256, 48) | (256, 64) | 2,330 MB | 3,180 MB | 1.33x |
| up_block_2 | `[1,240,2,424,2,128,48]` | (128, 48) | (128, 64) | 4,660 MB | 6,360 MB | 1.33x |

### Verification

The tt-swiss memory report for op 1298 reports a padded output of **11,520 MB**.
Our calculation: `6,039,797,760 elements * 2 bytes / (1024*1024) = 11,520.0 MB`.
**Exact match.**
