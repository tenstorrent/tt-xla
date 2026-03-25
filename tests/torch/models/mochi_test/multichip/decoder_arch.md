# Mochi VAE Decoder — Architecture & Memory Analysis

## Sources

- **HuggingFace Diffusers**: [autoencoder_kl_mochi.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_kl_mochi.py)
- **Genmo Original Repo**: [models.py](https://github.com/genmoai/mochi/blob/main/src/genmo/mochi_preview/vae/models.py)
- **Genmo Blog**: [Mochi 1: A new SOTA in open text-to-video](https://www.genmo.ai/blog/mochi-1-a-new-sota-in-open-text-to-video)

## 1. Overview

The Mochi VAE is an **Asymmetric VAE (AsymmVAE)** — the decoder is intentionally larger than the encoder (2x base channels: 128 vs 64). The decoder has **no attention layers** (encoder has attention at 4/5 stages), relying entirely on convolutional capacity for reconstruction quality.

### Compression Ratios


| Dimension | Ratio | Input (latent) | Output (video) |
| --------- | ----- | -------------- | -------------- |
| Temporal  | 6x    | 8              | 48             |
| Height    | 8x    | 60             | 480            |
| Width     | 8x    | 106            | 848            |
| Channels  | -     | 12             | 3              |


**Total volumetric compression: 384x** (8×8×6)

### Key Design Choices

- **No attention in decoder** — all capacity comes from conv blocks (19 ResNet blocks total)
- **No transposed convolutions** — upsampling via linear projection + depth-to-space-time (sub-pixel shuffle 3D)
- **Causal convolutions** — temporal padding only on left (past), never right (future)
- **~362M parameters** total VAE (decoder gets the larger share)

## 2. Class Hierarchy (HuggingFace Diffusers)


| Class                     | Lines      | Role                                                                |
| ------------------------- | ---------- | ------------------------------------------------------------------- |
| `AutoencoderKLMochi`      | L655-1105  | Top-level VAE; owns encoder + decoder; handles tiling               |
| `MochiDecoder3D`          | L548-652   | The decoder network                                                 |
| `MochiMidBlock3D`         | L245-326   | Block of N ResNet layers (used for block_in and block_out)          |
| `MochiUpBlock3D`          | L329-405   | N ResNet layers + linear unpatchify upsample                        |
| `MochiResnetBlock3D`      | L69-124    | Pre-norm residual: GN→SiLU→CausalConv3d→GN→SiLU→CausalConv3d + skip |
| `MochiChunkedGroupNorm3D` | L35-66     | Per-frame GroupNorm (32 groups, processes T frames in chunks)       |
| `CogVideoXCausalConv3d`   | (external) | Causal 3D conv: left-pad temporal, symmetric spatial, replicate pad |


## 3. Decoder Configuration

From `AutoencoderKLMochi.__init__()` (L736-744):

```python
MochiDecoder3D(
    in_channels=12,                          # latent channels
    out_channels=3,                          # RGB
    block_out_channels=(128, 256, 512, 768), # channel progression
    layers_per_block=(3, 3, 4, 6, 3),       # ResBlocks per stage
    temporal_expansions=(1, 2, 3),           # per upsample stage
    spatial_expansions=(2, 2, 2),            # per upsample stage
    act_fn="silu",
)
```

## 4. Layer-by-Layer Architecture

### Forward Pass Flow

```
Input: [B, 12, T, H, W]  (latent)
  │
  ▼
conv_in: Conv3d(12→768, k=1×1×1)           ← pointwise, no spatial change
  │
  ▼
block_in: 3× MochiResnetBlock3D(768)        ← MochiMidBlock3D, no attention
  │
  ▼
up_block_0: 6× ResBlock(768) + Unpatchify   ← 768→512, temporal ×3, spatial ×2
  │
  ▼
up_block_1: 4× ResBlock(512) + Unpatchify   ← 512→256, temporal ×2, spatial ×2
  │
  ▼
up_block_2: 3× ResBlock(256) + Unpatchify   ← 256→128, temporal ×1, spatial ×2
  │
  ▼
block_out: 3× MochiResnetBlock3D(128)       ← MochiMidBlock3D, no attention
  │
  ▼
SiLU → Linear(128→3)                        ← final projection to RGB
  │
  ▼
Output: [B, 3, T×6, H×8, W×8]
```

### Detailed Stage Breakdown

#### Stage 0: Input Projection

- **conv_in**: `nn.Conv3d(12, 768, kernel_size=(1,1,1))` (L584, L618)
- Simple channel expansion, no spatial/temporal change

#### Stage 1: block_in — MochiMidBlock3D (L585-589)

- **3× MochiResnetBlock3D(768)**, no attention
- Each ResBlock:
  1. `MochiChunkedGroupNorm3D(768)` — per-frame GN with 32 groups
  2. SiLU activation
  3. `CogVideoXCausalConv3d(768→768, k=3×3×3, stride=1)` — causal, replicate pad
  4. `MochiChunkedGroupNorm3D(768)`
  5. SiLU activation
  6. `CogVideoXCausalConv3d(768→768, k=3×3×3, stride=1)`
  7. Residual add: `output = conv2_out + input` (identity skip, no 1×1 conv)

#### Stage 2: up_block_0 — MochiUpBlock3D (L592-601, i=0)

- **6× MochiResnetBlock3D(768)** — same structure as Stage 1
- **Unpatchify Upsample** (L363, L391-403):
  1. `nn.Linear(768, 512 × 3 × 2 × 2)` = `nn.Linear(768, 6144)`
  2. Reshape: `[B, 6144, T, H, W]` → `[B, 512, T×3, H×2, W×2]`
  3. This is a 3D sub-pixel shuffle (depth-to-space-time)

#### Stage 3: up_block_1 — MochiUpBlock3D (i=1)

- **4× MochiResnetBlock3D(512)**
- **Unpatchify**: `nn.Linear(512, 256 × 2 × 2 × 2)` = `nn.Linear(512, 2048)`
  - Reshape → `[B, 256, T×2, H×2, W×2]`

#### Stage 4: up_block_2 — MochiUpBlock3D (i=2)

- **3× MochiResnetBlock3D(256)**
- **Unpatchify**: `nn.Linear(256, 128 × 1 × 2 × 2)` = `nn.Linear(256, 512)`
  - Reshape → `[B, 128, T×1, H×2, W×2]` (no temporal expansion at this stage)

#### Stage 5: block_out — MochiMidBlock3D (L603-607)

- **3× MochiResnetBlock3D(128)**, no attention
- Same ResBlock structure as Stage 1 but with 128 channels

#### Stage 6: Output Projection (L646-650)

1. SiLU activation
2. Permute to channel-last: `[B, T, H, W, 128]`
3. `proj_out`: `nn.Linear(128, 3)`
4. Permute back: `[B, 3, T, H, W]`

### ResBlock Count Summary


| Stage      | Channels | ResBlocks | Has Attention |
| ---------- | -------- | --------- | ------------- |
| block_in   | 768      | 3         | No            |
| up_block_0 | 768      | 6         | No            |
| up_block_1 | 512      | 4         | No            |
| up_block_2 | 256      | 3         | No            |
| block_out  | 128      | 3         | No            |
| **Total**  | —        | **19**    | **None**      |


## 5. Complete Shape & Memory Trace

Input: `[1, 12, 8, 60, 106]` → Output: `[1, 3, 48, 480, 848]`

### Activation Shapes at Each Stage


| #   | Layer                        | Output Shape               | Elements      | bfloat16    | float32  |
| --- | ---------------------------- | -------------------------- | ------------- | ----------- | -------- |
| 0   | INPUT (latent)               | [1, 12, 8, 60, 106]        | 610,560       | 1.2 MB      | 2.3 MB   |
| 1   | conv_in (12→768)             | [1, 768, 8, 60, 106]       | 39,075,840    | **74.5 MB** | 149.1 MB |
| 2   | block_in: 3× ResBlock(768)   | [1, 768, 8, 60, 106]       | 39,075,840    | 74.5 MB     | 149.1 MB |
| 3   | up_block_0: 6× ResBlock(768) | [1, 768, 8, 60, 106]       | 39,075,840    | 74.5 MB     | 149.1 MB |
| 4   | up_block_0/proj (768→6144)   | [1, 6144, 8, 60, 106]      | 312,606,720   | **596 MB**  | 1.16 GB  |
| 5   | up_block_0/unpatchify        | **[1, 512, 24, 120, 212]** | 312,606,720   | **596 MB**  | 1.16 GB  |
| 6   | up_block_1: 4× ResBlock(512) | [1, 512, 24, 120, 212]     | 312,606,720   | 596 MB      | 1.16 GB  |
| 7   | up_block_1/proj (512→2048)   | [1, 2048, 24, 120, 212]    | 1,250,426,880 | **2.33 GB** | 4.66 GB  |
| 8   | up_block_1/unpatchify        | **[1, 256, 48, 240, 424]** | 1,250,426,880 | **2.33 GB** | 4.66 GB  |
| 9   | up_block_2: 3× ResBlock(256) | [1, 256, 48, 240, 424]     | 1,250,426,880 | 2.33 GB     | 4.66 GB  |
| 10  | up_block_2/proj (256→512)    | [1, 512, 48, 240, 424]     | 2,500,853,760 | **4.66 GB** | 9.32 GB  |
| 11  | up_block_2/unpatchify        | **[1, 128, 48, 480, 848]** | 2,500,853,760 | **4.66 GB** | 9.32 GB  |
| 12  | block_out: 3× ResBlock(128)  | [1, 128, 48, 480, 848]     | 2,500,853,760 | 4.66 GB     | 9.32 GB  |
| 13  | SiLU                         | [1, 128, 48, 480, 848]     | 2,500,853,760 | 4.66 GB     | 9.32 GB  |
| 14  | proj_out (128→3) = OUTPUT    | [1, 3, 48, 480, 848]       | 58,613,760    | 112 MB      | 224 MB   |


### Peak Memory Bottlenecks (per-tensor)


| Rank | Location                        | Shape                   | bfloat16    | float32 |
| ---- | ------------------------------- | ----------------------- | ----------- | ------- |
| 1    | up_block_2/proj output          | [1, 512, 48, 240, 424]  | **4.66 GB** | 9.32 GB |
| 1    | block_out / up_block_2 output   | [1, 128, 48, 480, 848]  | **4.66 GB** | 9.32 GB |
| 3    | up_block_1/proj output          | [1, 2048, 24, 120, 212] | **2.33 GB** | 4.66 GB |
| 3    | up_block_2 ResBlock activations | [1, 256, 48, 240, 424]  | **2.33 GB** | 4.66 GB |
| 5    | up_block_0/proj output          | [1, 6144, 8, 60, 106]   | 596 MB      | 1.16 GB |


### Worst-Case Memory (two tensors alive simultaneously)

During convolution operations, both input and output (or padded intermediate) coexist:


| Location                    | Coexisting Tensors                               | Total bfloat16 | Total float32 |
| --------------------------- | ------------------------------------------------ | -------------- | ------------- |
| **block_out ResBlock conv** | [1,128,48,480,848] + [1,128,50,482,850] (padded) | **~9.5 GB**    | ~19 GB        |
| **up_block_2/proj**         | [1,256,48,240,424] + [1,512,48,240,424]          | **~7.0 GB**    | ~14 GB        |
| up_block_1/proj             | [1,512,24,120,212] + [1,2048,24,120,212]         | ~2.9 GB        | ~5.8 GB       |


### Causal Conv3d Padding Impact

Each `CogVideoXCausalConv3d(k=3, stride=1)` pads input to `[B, C, T+2, H+2, W+2]`:


| Stage                 | Channels | Padded Shape               | bfloat16    |
| --------------------- | -------- | -------------------------- | ----------- |
| block_in / up_block_0 | 768      | [1, 768, 10, 62, 108]      | 96 MB       |
| up_block_1            | 512      | [1, 512, 26, 122, 214]     | 647 MB      |
| up_block_2            | 256      | [1, 256, 50, 242, 426]     | **2.46 GB** |
| **block_out**         | 128      | **[1, 128, 50, 482, 850]** | **4.88 GB** |


## 6. Memory Budget Analysis (4 cards × 32 GB each)

### Total Available Memory

- **128 GB total** across 4 TT cards (32 GB each)
- After model weights (~362M params × 2 bytes ≈ 0.7 GB replicated), ~31.3 GB per card available for activations

### Why Sharding is Necessary (bfloat16)

- **Single card peak**: block_out ResBlock needs ~9.5 GB for just two coexisting tensors
- **With intermediates**: A single ResBlock in block_out needs input + padded_intermediate + output ≈ 14+ GB
- **Without sharding**: Single card (32 GB) could technically fit, but barely — no room for all the intermediate tensors in the conv chain
- **With 4-way sharding**: Each card handles ~2.3 GB peak per tensor → comfortable fit

### Weight Memory (replicated, bfloat16)


| Component                            | Approximate Size |
| ------------------------------------ | ---------------- |
| conv_in (12→768, 1×1×1)              | 18 KB            |
| ResBlocks (19 total, with GroupNorm) | ~680 MB          |
| Unpatchify linears (3 stages)        | ~15 MB           |
| proj_out (128→3)                     | <1 KB            |
| **Total decoder weights**            | **~700 MB**      |


## 7. Upsampling Mechanism Detail (Depth-to-Space-Time)

The unpatchify operation is a 3D extension of PixelShuffle:

```
Given: input [B, C_in, T, H, W]
1. Permute to channel-last: [B, T, H, W, C_in]
2. Linear projection: C_in → C_out × t_exp × s_exp × s_exp
3. Reshape: [B, T, H, W, C_out, t_exp, s_exp, s_exp]
4. Permute + reshape: [B, C_out, T×t_exp, H×s_exp, W×s_exp]
```

This avoids checkerboard artifacts of transposed convolutions and is more amenable to sharding (the linear projection can be column-partitioned across devices).

### Upsample Summary


| Stage      | In Channels | Out Channels | Temporal × | Spatial × | Linear Size |
| ---------- | ----------- | ------------ | ---------- | --------- | ----------- |
| up_block_0 | 768         | 512          | 3          | 2         | 768 → 6144  |
| up_block_1 | 512         | 256          | 2          | 2         | 512 → 2048  |
| up_block_2 | 256         | 128          | 1          | 2         | 256 → 512   |


## 8. Key Observations for Sharding Strategy

1. **Memory bottleneck is activations, not weights**: Weights are ~700 MB vs peak activation of ~4.66 GB per tensor in bfloat16.
2. **The last two stages dominate memory**: up_block_2 (256ch, 48×240×424) and block_out (128ch, 48×480×848) produce the largest activations.
3. **No attention = no KV cache concerns**: All 19 blocks are pure conv ResBlocks.
4. **Channel counts are all divisible by 4**: 768/4=192, 512/4=128, 256/4=64, 128/4=32 — all stages can be 4-way channel-sharded.
5. **Unpatchify linear projections** are easy to shard (standard linear layer).
6. **Causal convolutions** have asymmetric temporal padding (left-only) which may complicate spatial/temporal domain decomposition but does not affect channel sharding.
7. **GroupNorm with 32 groups**: With 4-way channel sharding, each device gets groups/4 groups (e.g., 32/4=8 groups for 768ch stage). GroupNorm is local to each shard when groups divide evenly.
8. **No skip connections between encoder and decoder**: Decoder is self-contained.
