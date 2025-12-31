# Mochi 1 Model Architecture - Deep Dive

## Table of Contents
1. [High-Level Overview](#high-level-overview)
2. [Asymmetric VAE (AsymmVAE)](#asymmetric-vae-asymmvae)
3. [Asymmetric Diffusion Transformer (AsymmDiT)](#asymmetric-diffusion-transformer-asymmdit)
4. [Text Conditioning Pipeline](#text-conditioning-pipeline)
5. [Attention Mechanism Deep Dive](#attention-mechanism-deep-dive)
6. [Training and Inference Flow](#training-and-inference-flow)
7. [LoRA Integration](#lora-integration)

---

## High-Level Overview

Mochi 1 is a 10B parameter video generation model using a latent diffusion architecture. The key innovation is the **Asymmetric Diffusion Transformer (AsymmDiT)** which processes visual and text tokens with asymmetric hidden dimensions.

```
┌─────────────────────────────────────────────────────────────────┐
│                     MOCHI 1 FULL PIPELINE                       │
└─────────────────────────────────────────────────────────────────┘

INPUT: Text Prompt + Random Noise
  │
  ├─────────────────────────────────────────────────────┐
  │                                                      │
  ▼                                                      ▼
┌──────────────────┐                          ┌──────────────────┐
│  T5-XXL Encoder  │                          │  Initial Noise   │
│   (text only)    │                          │   z ~ N(0, I)    │
│  google/t5-xxl   │                          │ [B,12,t,h,w]     │
└────────┬─────────┘                          └─────────┬────────┘
         │                                               │
         │ [B, 256, 4096]                               │
         ▼                                               │
┌─────────────────────────────────────────────────────┐ │
│           AsymmDiT (10B params)                     │ │
│  ┌───────────────────────────────────────────────┐  │ │
│  │  Conditioning:                                │  │ │
│  │  - Timestep Embedding: t → 3072-dim          │◄─┘
│  │  - T5 Attention Pool: 4096 → 3072-dim        │◄───┘
│  │  - T5 Dense Proj: 4096 → 1536-dim            │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Visual Path (3072-dim):                     │  │
│  │  - Patch Embedding (2x2)                      │  │
│  │  - 48 AsymmetricJointBlocks                   │  │
│  │  - MLP ratio: 8.0 (12,288 hidden)            │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Text Path (1536-dim):                       │  │
│  │  - Dense features from T5                     │  │
│  │  - 47 blocks (last block doesn't update)     │  │
│  │  - MLP ratio: 4.0 (6,144 hidden)             │  │
│  └───────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────┐  │
│  │  Joint Attention:                            │  │
│  │  - Non-square QKV projections                 │  │
│  │  - Text: 1536 → 3*3072 (upproject to visual) │  │
│  │  - Visual: 3072 → 3*3072                      │  │
│  │  - Concat and attend jointly                  │  │
│  │  - Separate output projs back to dimensions   │  │
│  └───────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼ [B, 12, t, h, w]
          ┌───────────────────────┐
          │  AsymmVAE Decoder     │
          │   (362M params)       │
          │  Decompress 128x:     │
          │  - 6x temporal        │
          │  - 8x8 spatial        │
          └──────────┬────────────┘
                     │
                     ▼
            [B, 3, T, H, W]
           OUTPUT VIDEO FRAMES
```

**Key Metrics:**
- Total parameters: ~10.4B (10B DiT + 362M VAE)
- Visual tokens: N = T × (H/8) × (W/8) / 4 = ~44,520 for 480p
- Text tokens: L = 256 (fixed)
- Visual dim: 3072, Text dim: 1536 (2:1 ratio)
- Heads: 24, Head dim: 128

---

## Asymmetric VAE (AsymmVAE)

The AsymmVAE performs **128x compression** (8×8 spatial, 6× temporal) to a 12-channel latent space.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ENCODER (not used in inference)         │
└─────────────────────────────────────────────────────────────┘

Video [B, 3, T, H, W]
  │
  ├─ Conv3d 3→64 (1×1×1)
  │
  ├─ ResBlocks (64 ch) × 3
  │
  ├─ DownsampleBlock: 64→128, reduce by (1, 2, 2)
  ├─ ResBlocks (128 ch) × 3
  │
  ├─ DownsampleBlock: 128→256, reduce by (2, 2, 2)  ← 1st temporal
  ├─ ResBlocks (256 ch) × 4
  │
  ├─ DownsampleBlock: 256→384, reduce by (3, 2, 2)  ← 2nd temporal
  ├─ ResBlocks (384 ch) × 6 + attention
  │
  ├─ ResBlocks (384 ch) × 3 + attention
  │
  ├─ RMSNorm + SiLU + Conv1x1 → [B, 24, t, h, w]
  │
  └─ Split: mean [B, 12, t, h, w], logvar [B, 12, t, h, w]

Total reduction: (1×2×3) = 6x temporal, (2×2×2) = 8x spatial

┌─────────────────────────────────────────────────────────────┐
│                     DECODER (used in inference)             │
└─────────────────────────────────────────────────────────────┘

Latents [B, 12, t, h, w]  (normalized to [-1, 1])
  │
  ├─ Conv3d 12→768 (1×1×1)
  ├─ ResBlocks (768 ch) × 6
  │
  ├─ CausalUpsampleBlock: 768→384, expand by (3, 2, 2)
  │   ├─ ResBlocks (768) × 3
  │   ├─ Conv1x1: 768 → 384×3×4 = 4608
  │   └─ DepthToSpaceTime: rearrange to [B, 384, t×3, h×2, w×2]
  │
  ├─ CausalUpsampleBlock: 384→256, expand by (2, 2, 2)
  │   └─ [similar structure]
  │
  ├─ CausalUpsampleBlock: 256→128, expand by (1, 2, 2)
  │   └─ [similar structure]
  │
  ├─ ResBlocks (128 ch) × 3
  │
  ├─ SiLU + Conv1x1: 128 → 3
  │
  └─ Output: [B, 3, T, H, W]

Total expansion: (3×2×1) = 6x temporal, (2×2×2) = 8x spatial
```

**Important Details:**

1. **Causal Padding**: All temporal convolutions use causal padding (pad front, not back) to maintain autoregressive structure
2. **DepthToSpaceTime**: Upsampling via channel-to-space rearrangement (no learnable params, just reshaping)
3. **Context Parallel**: Supports distributed inference by splitting along temporal dimension
4. **Asymmetry**:
   - Encoder: ~100-120M params (64 base channels, max 384 channels)
   - Decoder: ~240-260M params (128 base channels, starts at 768 channels)
   - **Decoder is 2-2.5× larger** - decoding (generation) is harder than encoding (compression)
   - Decoder needs to "hallucinate" fine details from compressed latents
5. **VAE Training**: Trained completely separately, **unsupervised** (no captions/labels needed!)
   - Only needs raw video data
   - Loss: Reconstruction + KL divergence
   - Frozen forever after pre-training

### Latent Space Statistics

```python
# VAE latents are normalized for DiT:
dit_latents = (vae_latents - vae_mean) / vae_std

vae_mean = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
vae_std = [0.925, 0.934, 0.946, 0.939, 0.961, 1.033, 0.979, 1.024, 0.983, 1.046, 0.964, 1.004]
```

---

## Asymmetric Diffusion Transformer (AsymmDiT)

### Model Specifications

```
┌──────────────────────────────────────────────────────────────┐
│                   AsymmDiT Configuration                     │
├──────────────────────────────────────────────────────────────┤
│ Total Depth:           48 layers                             │
│ Num Heads:             24                                    │
│ Head Dim:              128                                   │
│ Visual Hidden:         3072 (24 × 128)                       │
│ Text Hidden:           1536                                  │
│ Visual MLP Hidden:     12,288 (3072 × 4)                     │
│ Text MLP Hidden:       6,144 (1536 × 4)                      │
│ Patch Size:            2×2                                   │
│ Input Channels:        12 (latent space)                     │
│ T5 Token Length:       256                                   │
│ T5 Feature Dim:        4096                                  │
└──────────────────────────────────────────────────────────────┘
```

### Forward Pass Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    AsymmDiT Forward Pass                       │
└────────────────────────────────────────────────────────────────┘

INPUT:
  x: [B, 12, T, H, W]          (noisy latents)
  sigma: [B]                    (noise level, 0=clean, 1=pure noise)
  y_feat: [B, 256, 4096]       (T5 features)
  y_mask: [B, 256]             (attention mask for T5)

┌──────────────────────────────────────────────────────────────┐
│ STEP 1: PREPARE EMBEDDINGS                                   │
└──────────────────────────────────────────────────────────────┘

  ┌─ Visual Embedding ─────────────────────────────────────┐
  │ x [B,12,T,H,W] → PatchEmbed(2×2)                      │
  │   → [B, N, 3072] where N = T×(H/2)×(W/2)              │
  │                                                         │
  │ Position Matrix [N, 3]:                                │
  │   pos[i] = [frame_idx, row_idx, col_idx]              │
  │                                                         │
  │ RoPE Frequencies [3, 24, 64]:                          │
  │   Mixed rotation: temporal + spatial                   │
  │   rope_cos, rope_sin = compute_mixed_rotation(...)    │
  └─────────────────────────────────────────────────────────┘

  ┌─ Text Embedding ───────────────────────────────────────┐
  │ y_feat [B, 256, 4096]                                  │
  │   → T5 Attention Pool → [B, 3072]  (for conditioning)  │
  │   → T5 Dense Proj → [B, 256, 1536] (for joint attn)   │
  └─────────────────────────────────────────────────────────┘

  ┌─ Conditioning Vector ──────────────────────────────────┐
  │ c_t = TimestepEmbedder(1 - sigma)  → [B, 3072]        │
  │ c_text = AttentionPool(y_feat, y_mask) → [B, 3072]    │
  │ c = c_t + c_text  → [B, 3072]                          │
  └─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│ STEP 2: PROCESS THROUGH 48 AsymmetricJointBlocks            │
└──────────────────────────────────────────────────────────────┘

for i in range(48):
    x, y = AsymmetricJointBlock(
        x=[B, N, 3072],    # visual tokens
        y=[B, 256, 1536],  # text tokens
        c=[B, 3072],       # conditioning
        rope_cos, rope_sin,
        update_y=(i < 47)  # last block doesn't update text
    )

┌──────────────────────────────────────────────────────────────┐
│ STEP 3: FINAL LAYER                                          │
└──────────────────────────────────────────────────────────────┘

  x [B, N, 3072]
    → LayerNorm (modulated by c)
    → Linear: 3072 → 2×2×12 = 48
    → Rearrange: [B, N, 48] → [B, 12, T, H, W]

OUTPUT: Predicted noise or v-prediction [B, 12, T, H, W]
```

---

## Asymmetric Joint Block

Each block performs joint attention over visual and text tokens, but with **separate MLPs** and **asymmetric dimensions**.

```
┌────────────────────────────────────────────────────────────────┐
│             AsymmetricJointBlock Architecture                  │
└────────────────────────────────────────────────────────────────┘

INPUT:
  x: [B, N, 3072]      visual tokens
  y: [B, 256, 1536]    text tokens
  c: [B, 3072]         conditioning vector

┌─────────────────────────────────────────────────────────────┐
│ MODULATION FROM CONDITIONING                                │
└─────────────────────────────────────────────────────────────┘

c → SiLU → Linear(3072 → 4×3072)
  ├─ scale_msa_x [B, 3072]   (for visual pre-attn norm)
  ├─ gate_msa_x  [B, 3072]   (for visual attn gating)
  ├─ scale_mlp_x [B, 3072]   (for visual pre-MLP norm)
  └─ gate_mlp_x  [B, 3072]   (for visual MLP gating)

c → SiLU → Linear(3072 → 4×1536)  [if update_y=True]
  ├─ scale_msa_y [B, 1536]   (for text pre-attn norm)
  ├─ gate_msa_y  [B, 1536]   (for text attn gating)
  ├─ scale_mlp_y [B, 1536]   (for text pre-MLP norm)
  └─ gate_mlp_y  [B, 1536]   (for text MLP gating)

┌─────────────────────────────────────────────────────────────┐
│ ASYMMETRIC ATTENTION                                        │
└─────────────────────────────────────────────────────────────┘

  ┌─ Pre-Norm (Modulated RMSNorm) ─────────────────────────┐
  │ x_norm = modulated_rmsnorm(x, scale_msa_x)             │
  │ y_norm = modulated_rmsnorm(y, scale_msa_y)             │
  └─────────────────────────────────────────────────────────┘

  ┌─ QKV Projections ───────────────────────────────────────┐
  │ qkv_x = LoraLinear(3072 → 3×3072)(x_norm)              │
  │ qkv_y = LoraLinear(1536 → 3×3072)(y_norm)  ← upproject!│
  │                                                          │
  │ Split & normalize:                                       │
  │   q_x, k_x, v_x = split(qkv_x)  [B, N, 24, 128]        │
  │   q_y, k_y, v_y = split(qkv_y)  [B, 256, 24, 128]      │
  │                                                          │
  │   q_x, k_x = RMSNorm(head_dim), RoPE                    │
  │   q_y, k_y = RMSNorm(head_dim)  (no RoPE for text)     │
  └──────────────────────────────────────────────────────────┘

  ┌─ Concatenate & Pack ────────────────────────────────────┐
  │ q = cat([q_x, q_y], dim=1)  [B, N+256, 24, 128]        │
  │ k = cat([k_x, k_y], dim=1)  [B, N+256, 24, 128]        │
  │ v = cat([v_x, v_y], dim=1)  [B, N+256, 24, 128]        │
  │                                                          │
  │ Pack for flash attention (remove padding):               │
  │   valid_indices, cu_seqlens, max_seqlen                 │
  └──────────────────────────────────────────────────────────┘

  ┌─ Flash Attention ───────────────────────────────────────┐
  │ out = flash_varlen_attn(                                │
  │     q, k, v,                                            │
  │     cu_seqlens_q=cu_seqlens,                            │
  │     cu_seqlens_k=cu_seqlens,                            │
  │     max_seqlen_q=max_seqlen,                            │
  │     max_seqlen_k=max_seqlen                             │
  │ )  → [total_valid_tokens, 24, 128]                      │
  └──────────────────────────────────────────────────────────┘

  ┌─ Split & Output Projection ─────────────────────────────┐
  │ out_x, out_y = split(out, [N, 256])                     │
  │                                                          │
  │ x_attn = LoraLinear(3072 → 3072)(out_x)                │
  │ y_attn = LoraLinear(3072 → 1536)(out_y)  ← downproject!│
  └──────────────────────────────────────────────────────────┘

  ┌─ Residual + Gated RMSNorm ──────────────────────────────┐
  │ x = residual_tanh_gated_rmsnorm(x, x_attn, gate_msa_x) │
  │   = x + tanh(gate_msa_x) * RMSNorm(x_attn)              │
  │                                                          │
  │ y = residual_tanh_gated_rmsnorm(y, y_attn, gate_msa_y) │
  └──────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ FEEDFORWARD (MLP)                                           │
└─────────────────────────────────────────────────────────────┘

  ┌─ Visual MLP ────────────────────────────────────────────┐
  │ x_norm = modulated_rmsnorm(x, scale_mlp_x)              │
  │ x_ff = FeedForward(3072 → 12288 → 3072)(x_norm)        │
  │ x = residual_tanh_gated_rmsnorm(x, x_ff, gate_mlp_x)   │
  └──────────────────────────────────────────────────────────┘

  ┌─ Text MLP [if update_y] ────────────────────────────────┐
  │ y_norm = modulated_rmsnorm(y, scale_mlp_y)              │
  │ y_ff = FeedForward(1536 → 6144 → 1536)(y_norm)         │
  │ y = residual_tanh_gated_rmsnorm(y, y_ff, gate_mlp_y)   │
  └──────────────────────────────────────────────────────────┘

OUTPUT:
  x: [B, N, 3072]      updated visual tokens
  y: [B, 256, 1536]    updated text tokens (if update_y=True)
```

### Key Design Decisions

1. **Non-square QKV projections**: Text (1536) → 3×3072, Visual (3072) → 3×3072
   - Unifies both modalities in the same attention space
   - Allows joint attention computation

2. **Asymmetric MLPs**: Visual gets 8x expansion, text gets 4x expansion
   - Visual: 3072 → 12,288 → 3072 (75M params per layer)
   - Text: 1536 → 6,144 → 1536 (19M params per layer)
   - **~5× more parameters for visual reasoning** (due to quadratic MLP scaling)

3. **Last block optimization**: Block 47 (last) doesn't update text
   - Text features not needed for final visual prediction
   - Saves computation (~19M FLOPs per forward)

4. **Modulated normalization**: All norms are conditioned on timestep + caption
   - Similar to AdaLN in DiT, but using RMSNorm instead of LayerNorm
   - Separate scales/gates for attention and MLP

### Non-Square Projection Deep Dive

**Why upproject text from 1536 to 3072 instead of keeping everything at the same dimension?**

```
┌────────────────────────────────────────────────────────────────┐
│              The Non-Square Projection Strategy                │
└────────────────────────────────────────────────────────────────┘

VISUAL TOKENS:  [B, ~50,000, 3072]   ← Native dimension
TEXT TOKENS:    [B, 256, 1536]       ← Half dimension

Goal: Joint attention in a unified space

┌─ Option 1: Unified 3072 (NOT USED) ────────────────────────┐
│ Keep text at 3072 throughout                               │
│                                                             │
│ Text MLP params per layer:                                 │
│   W_in:  3072 × (3072×4) = 37.7M                          │
│   W_out: (3072×4) × 3072 = 37.7M                          │
│   Total: 75.4M params × 48 layers = 3.6B params           │
│                                                             │
│ Problem: Text doesn't need that much capacity!             │
│   - Only 256 tokens vs 50,000 visual                       │
│   - Simpler semantic structure                             │
│   - Massive parameter waste                                │
└─────────────────────────────────────────────────────────────┘

┌─ Option 2: Unified 1536 (NOT USED) ────────────────────────┐
│ Downproject visual to 1536                                  │
│                                                             │
│ Problem 1: Visual path loses capacity                       │
│   - Need to downproject 50K tokens: 50K × 3072 × 1536      │
│   - That's 236 GFLOPs just for projection!                 │
│                                                             │
│ Problem 2: Visual information loss                          │
│   - Spatial-temporal complexity needs high dimension        │
│   - Compressing 50K tokens hurts quality                   │
└─────────────────────────────────────────────────────────────┘

┌─ Option 3: Asymmetric with Upproject (USED!) ──────────────┐
│ Keep native dimensions, upproject text only for attention   │
│                                                             │
│ Text stays at 1536 for MLPs:                               │
│   W_in:  1536 × (1536×4) = 9.4M                           │
│   W_out: (1536×4) × 1536 = 9.4M                           │
│   Total: 18.9M params × 48 layers = 907M params           │
│   Savings: 3.6B - 0.9B = 2.7B params saved!               │
│                                                             │
│ Text upprojects ONLY for QKV (cheap!):                     │
│   qkv_y: 1536 → 3×3072 = 14.2M params                     │
│   Only 256 tokens × 1536 × 9216 = 3.6 GFLOPs              │
│   vs downproject visual: 50K × 3072 × 1536 = 236 GFLOPs   │
│   65× more efficient!                                      │
│                                                             │
│ Benefits:                                                   │
│   ✓ Visual path keeps full 3072-dim capacity               │
│   ✓ Text path saves 2.7B params                            │
│   ✓ Joint attention in unified 3072-dim space              │
│   ✓ Minimal projection cost (only 256 tokens)              │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step: How Non-Square Projections Work

```
┌────────────────────────────────────────────────────────────────┐
│              Forward Pass with Non-Square Projections          │
└────────────────────────────────────────────────────────────────┘

INPUT:
  x: [B, 50000, 3072]  visual tokens
  y: [B, 256, 1536]    text tokens

STEP 1: QKV Projections (non-square!)
  ┌─────────────────────────────────────────────────────────┐
  │ Visual Path:                                            │
  │   qkv_x = Linear(3072 → 9216)(x)                       │
  │   qkv_x = [B, 50000, 9216]                             │
  │   q_x, k_x, v_x = split(qkv_x, dim=-1)                │
  │   → [B, 50000, 3072] each                              │
  │   → reshape: [B, 50000, 24, 128]                       │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │ Text Path (NON-SQUARE!):                                │
  │   qkv_y = Linear(1536 → 9216)(y)  ← 1536 to 9216!     │
  │   qkv_y = [B, 256, 9216]                               │
  │   q_y, k_y, v_y = split(qkv_y, dim=-1)                │
  │   → [B, 256, 3072] each  ← Now same as visual!        │
  │   → reshape: [B, 256, 24, 128]                         │
  └─────────────────────────────────────────────────────────┘

STEP 2: Concatenate in Unified Space
  ┌─────────────────────────────────────────────────────────┐
  │ q = cat([q_x, q_y], dim=1)  [B, 50256, 24, 128]       │
  │ k = cat([k_x, k_y], dim=1)  [B, 50256, 24, 128]       │
  │ v = cat([v_x, v_y], dim=1)  [B, 50256, 24, 128]       │
  │                                                         │
  │ All in same space - can attend jointly!                │
  └─────────────────────────────────────────────────────────┘

STEP 3: Joint Attention
  ┌─────────────────────────────────────────────────────────┐
  │ out = flash_attention(q, k, v)                          │
  │ out = [total_valid_tokens, 24, 128]                     │
  │                                                         │
  │ Cross-modal attention naturally happens:                │
  │   - Text tokens attend to visual tokens                 │
  │   - Visual tokens attend to text tokens                 │
  │   - Same head_dim=128 for all                          │
  └─────────────────────────────────────────────────────────┘

STEP 4: Split and Output Projection
  ┌─────────────────────────────────────────────────────────┐
  │ Split back:                                             │
  │   out_x = out[:50000]  [B, 50000, 24, 128]            │
  │   out_y = out[50000:]  [B, 256, 24, 128]              │
  │                                                         │
  │ Reshape:                                                │
  │   out_x → [B, 50000, 3072]                             │
  │   out_y → [B, 256, 3072]  ← Still in unified space    │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │ Visual Output (square):                                 │
  │   x_out = Linear(3072 → 3072)(out_x)                   │
  │   x_out = [B, 50000, 3072]  ← Stays at 3072           │
  └─────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────┐
  │ Text Output (NON-SQUARE!):                              │
  │   y_out = Linear(3072 → 1536)(out_y)  ← 3072 to 1536! │
  │   y_out = [B, 256, 1536]  ← Back to native dim        │
  └─────────────────────────────────────────────────────────┘

RESULT:
  x: [B, 50000, 3072]  ← visual stays in its native dim
  y: [B, 256, 1536]    ← text back to its native dim
```

### Parameter Breakdown: Why ~5× Not 4×?

```
┌────────────────────────────────────────────────────────────────┐
│        Parameter Count per AsymmetricJointBlock                │
└────────────────────────────────────────────────────────────────┘

VISUAL PATH (3072-dim):
  ┌─ Attention ─────────────────────────────────────────────┐
  │ qkv_x:   3072 × 9216  = 28.3M                          │
  │ proj_x:  3072 × 3072  = 9.4M                           │
  │ Subtotal: 37.7M                                         │
  └─────────────────────────────────────────────────────────┘

  ┌─ MLP ───────────────────────────────────────────────────┐
  │ fc_in:   3072 × 12288 = 37.7M                          │
  │ fc_out:  12288 × 3072 = 37.7M                          │
  │ Subtotal: 75.4M                                         │
  └─────────────────────────────────────────────────────────┘

  ┌─ Modulation ────────────────────────────────────────────┐
  │ adaLN:   3072 × (4×3072) = 37.7M                       │
  │ Subtotal: 37.7M                                         │
  └─────────────────────────────────────────────────────────┘

  VISUAL TOTAL: 37.7 + 75.4 + 37.7 = ~151M params

TEXT PATH (1536-dim):
  ┌─ Attention ─────────────────────────────────────────────┐
  │ qkv_y:   1536 × 9216  = 14.2M  ← non-square!          │
  │ proj_y:  3072 × 1536  = 4.7M   ← non-square!          │
  │ Subtotal: 18.9M                                         │
  └─────────────────────────────────────────────────────────┘

  ┌─ MLP ───────────────────────────────────────────────────┐
  │ fc_in:   1536 × 6144  = 9.4M                           │
  │ fc_out:  6144 × 1536  = 9.4M                           │
  │ Subtotal: 18.9M                                         │
  └─────────────────────────────────────────────────────────┘

  ┌─ Modulation ────────────────────────────────────────────┐
  │ adaLN:   3072 × (4×1536) = 18.9M                       │
  │ Subtotal: 18.9M                                         │
  └─────────────────────────────────────────────────────────┘

  TEXT TOTAL: 18.9 + 18.9 + 18.9 = ~57M params

┌────────────────────────────────────────────────────────────┐
│ RATIO: 151M / 57M = 2.65×                                 │
│                                                            │
│ Wait, where's the ~5×?                                    │
│                                                            │
│ The ~5× comes from MLP-only comparison:                   │
│   Visual MLP: 75.4M                                       │
│   Text MLP:   18.9M                                       │
│   Ratio: 75.4 / 18.9 = 3.99× ≈ 4×                        │
│                                                            │
│ But MLPs dominate parameter count!                        │
│   Visual: 75.4M MLP / 151M total = 50%                   │
│   Text:   18.9M MLP / 57M total = 33%                    │
│                                                            │
│ Key insight: MLP parameters scale QUADRATICALLY:          │
│   dim²  × mlp_ratio² = (2×)² × (2×) = 8×                │
│   BUT: visual uses ratio=8, text uses ratio=4            │
│   So: (3072/1536)² × (8/4) = 4 × 2 = 8×                  │
│                                                            │
│ Weighted average across all components: ~2.65×            │
│ But the COMPUTE is dominated by visual due to token count│
└────────────────────────────────────────────────────────────┘
```

---

## Text Conditioning Pipeline

**CRITICAL UNDERSTANDING**: Text serves TWO completely separate roles in Mochi! Don't confuse them.

```
┌────────────────────────────────────────────────────────────────┐
│              T5-XXL Text Encoding Pipeline                     │
└────────────────────────────────────────────────────────────────┘

Text Prompt: "A cat playing piano"
  │
  ├─ Tokenizer (max_length=256)
  │
  ▼
Token IDs: [1, 71, 1712, 1556, 11295, 1, 0, 0, ..., 0]
Attention Mask: [1, 1, 1, 1, 1, 1, 0, 0, ..., 0]
  │
  ├─ T5EncoderModel (google/t5-v1_1-xxl)
  │   - 24 layers, 4096 hidden dim
  │   - FSDP wrapped for multi-GPU
  │
  ▼
T5 Features: [B, 256, 4096]
  │
  ├───────────────────────────────┬──────────────────────────────┐
  │                               │                              │
  │ ROLE 1: Global Conditioning   │  ROLE 2: Dense Token Feats  │
  ▼                               ▼                              │
┌──────────────────────────┐  ┌──────────────────────────┐     │
│  AttentionPool           │  │  Linear Projection       │     │
│  (4096 → 3072)           │  │  (4096 → 1536)          │     │
│  8 heads, cross-attn     │  │  Per-token features      │     │
│  Learnable query         │  │  No pooling              │     │
└──────────┬───────────────┘  └──────────┬───────────────┘     │
           │                             │                      │
           ▼ [B, 3072]                   ▼ [B, 256, 1536]      │
           │                             │                      │
           │ Single vector               │ 256 separate         │
           │ "What to generate"          │ "Semantic features"  │
           │                             │                      │
      ┌────┴─────┐                       │                      │
      │ Timestep │                       │                      │
      │ Embedding│                       │                      │
      └────┬─────┘                       │                      │
           │                             │                      │
           ▼ c [B, 3072]                 │                      │
      ┌─────────────────┐                │                      │
      │  Modulates ALL  │                │                      │
      │  LayerNorms in  │                │                      │
      │  48 blocks      │                │                      │
      │  (scale & gate) │                │                      │
      └─────────────────┘                │                      │
           │                             │                      │
           └──────────────► AsymmetricJointBlocks ◄─────────────┘
                                         │
                                    Joint Attention
                                    with visual tokens
```

### The Two Roles Explained

#### Role 1: Global Conditioning Vector (3072-dim)

```
T5 Features [B, 256, 4096]
    │
    └─► AttentionPool (learnable query attends to all tokens)
            │
            ▼ [B, 3072] ← Single vector per sample
            │
            ├─► Modulate scale_msa (pre-attention norm)
            ├─► Modulate gate_msa (post-attention gate)
            ├─► Modulate scale_mlp (pre-MLP norm)
            └─► Modulate gate_mlp (post-MLP gate)

Purpose: High-level semantic control
  - "What should this image/video be about?"
  - Controls feature magnitude via adaptive normalization
  - Combined with timestep embedding: c = f(timestep) + f(caption)
  - Influences ALL 48 layers through modulation

Dimension: 3072 (matches visual hidden dim)
  - MUST match visual dimension for modulation compatibility
  - Used in: c → Linear(3072 → 4×3072) → scales and gates
```

#### Role 2: Dense Token Features (1536-dim)

```
T5 Features [B, 256, 4096]
    │
    └─► Linear Projection (simple matrix multiply, no pooling)
            │
            ▼ [B, 256, 1536] ← 256 separate token vectors
            │
            └─► Participate in joint attention with visual tokens
                  - Upproject to 3072 for Q, K, V
                  - Attend jointly with ~50K visual tokens
                  - Enable fine-grained text-visual alignment

Purpose: Fine-grained semantic features
  - "A cat [token 1] playing [token 2] piano [token 3]"
  - Each token provides specific semantic information
  - Enables cross-modal attention between text and visual
  - Each text token can attend to relevant visual regions

Dimension: 1536 (half of visual dimension)
  - Efficiency: Text is "simpler" than visual (fewer concepts)
  - 256 text tokens << 50,000 visual tokens
  - Non-square projection: 1536 → 3072 for joint attention
```

### Why Two Separate Dimensions?

```
┌────────────────────────────────────────────────────────────────┐
│                 Design Rationale                               │
└────────────────────────────────────────────────────────────────┘

Global Conditioning (3072):
  ✓ MUST be 3072 to match visual dimension
  ✓ Used for modulation: c → Linear(3072 → 4×3072)
  ✓ If it were 1536, would need extra projection layer
  ✓ Single vector, so parameter efficiency not a concern

Dense Tokens (1536):
  ✓ CAN be smaller - text is semantically simpler than visual
  ✓ 256 tokens, so 2× reduction = 2× param savings
  ✓ Parameter efficiency:
      - Text path: 1536-dim → ~39M params/block
      - Visual path: 3072-dim → ~190M params/block
      - ~5× fewer params for text (justified by complexity)
  ✓ Upproject only 256 tokens (cheap) vs downproject 50K tokens (expensive)

This is NOT arbitrary - it's a carefully optimized design!
```

### Attention Pool Details

```python
class AttentionPool(nn.Module):
    """
    Learnable query vector attends to T5 token features.
    Compresses variable-length caption to fixed global vector.
    Similar to CLIP's attention pooling.
    """
    def __init__(self, feat_dim=4096, num_heads=8, output_dim=3072):
        # Learnable query - learns to extract most relevant info
        self.query = nn.Parameter(torch.randn(1, 1, feat_dim))
        self.attn = nn.MultiheadAttention(feat_dim, num_heads)
        self.proj = nn.Linear(feat_dim, output_dim)

    def forward(self, x, mask):
        # x: [B, 256, 4096], mask: [B, 256]
        # query: [1, 1, 4096] → broadcast to [B, 1, 4096]
        # Cross-attention: query attends to all valid tokens
        pooled = self.attn(self.query, x, x, key_padding_mask=~mask)
        return self.proj(pooled).squeeze(1)  # [B, 3072]
```

---

## Attention Mechanism Deep Dive

### Flash Attention with Packed Sequences

Mochi uses Flash Attention with variable-length packing to efficiently handle text padding.

```
┌────────────────────────────────────────────────────────────────┐
│              Packed Sequence Format (Flash Attention)          │
└────────────────────────────────────────────────────────────────┘

Example: Batch of 2, Visual tokens N=100, Max text length=256

Sample 1: 100 visual + 180 text (76 padding)
Sample 2: 100 visual + 200 text (56 padding)

┌─ Unpacked Format (wasteful) ────────────────────────────────┐
│ Sample 1: [v0, v1, ..., v99, t0, t1, ..., t179, PAD×76]    │
│ Sample 2: [v0, v1, ..., v99, t0, t1, ..., t199, PAD×56]    │
│                                                              │
│ Total tokens: 2 × 356 = 712                                 │
│ Valid tokens: 280 + 300 = 580                               │
│ Padding waste: 132 tokens (18.5%)                           │
└──────────────────────────────────────────────────────────────┘

┌─ Packed Format (efficient) ─────────────────────────────────┐
│ [v0_1, ..., v99_1, t0_1, ..., t179_1,                      │
│  v0_2, ..., v99_2, t0_2, ..., t199_2]                      │
│                                                              │
│ Total tokens: 580 (no padding!)                             │
│                                                              │
│ cu_seqlens: [0, 280, 580]  (cumulative sequence lengths)    │
│ max_seqlen: 300                                             │
│ valid_token_indices: [0, 1, ..., 579]  (all non-padding)   │
└──────────────────────────────────────────────────────────────┘

Compute Packed Indices:
  mask = F.pad(text_mask, (num_visual_tokens, 0), value=True)
  valid_token_indices = torch.nonzero(mask.flatten())
  cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0), (1, 0))
```

### RoPE (Rotary Position Embedding)

Mochi uses **mixed RoPE** for 3D (temporal + spatial) position encoding.

```
┌────────────────────────────────────────────────────────────────┐
│                    Mixed 3D RoPE                               │
└────────────────────────────────────────────────────────────────┘

Position Matrix [N, 3]:
  N = T × H × W  (after patch embedding)
  pos[i] = [frame_idx, height_idx, width_idx]

Frequency Matrix [3, num_heads, head_dim/2]:
  freqs[0] = temporal frequencies
  freqs[1] = height frequencies
  freqs[2] = width frequencies

Compute rotation:
  theta = pos @ freqs  # [N, num_heads, head_dim/2]
  rope_cos = cos(theta)
  rope_sin = sin(theta)

Apply to Q, K (not V):
  q_rotated = apply_rotary_emb_qk_real(q, rope_cos, rope_sin)
  k_rotated = apply_rotary_emb_qk_real(k, rope_cos, rope_sin)

┌─ Rotation Formula (for complex numbers) ────────────────────┐
│ Split head_dim into pairs: [x0, x1, x2, x3, ...] →         │
│   pairs: [(x0, x1), (x2, x3), ...]                         │
│                                                              │
│ Rotate each pair:                                            │
│   x0_new = x0 * cos(θ) - x1 * sin(θ)                       │
│   x1_new = x0 * sin(θ) + x1 * cos(θ)                       │
└──────────────────────────────────────────────────────────────┘

Example dimensions:
  T=31, H=60, W=106 (for 480p video)
  After 2×2 patches: T=31, pH=30, pW=53
  N = 31 × 30 × 53 = 49,290 visual tokens
```

### Attention Math

```
Standard Multi-Head Attention (per head):

Q = q_norm_x(Linear(X)) + q_norm_y(Linear(Y))  [total_tokens, head_dim]
K = k_norm_x(Linear(X)) + k_norm_y(Linear(Y))  [total_tokens, head_dim]
V = Linear(X) + Linear(Y)                       [total_tokens, head_dim]

Attention(Q, K, V) = softmax(Q @ K^T / √d) @ V

Output split:
  out_x = out[:N]    → Linear(3072 → 3072)
  out_y = out[N:]    → Linear(3072 → 1536)
```

---

## Training and Inference Flow

### Three-Stage Training Pipeline

**CRITICAL**: Mochi is trained in THREE separate stages. Understanding this is key to understanding what's frozen vs trainable.

```
┌────────────────────────────────────────────────────────────────┐
│                   STAGE 1: VAE Pre-Training                    │
│                      (Completely Separate)                     │
└────────────────────────────────────────────────────────────────┘

Input: Raw video data ONLY (no captions, no labels!)
Goal: Learn to compress video to latent space

┌─────────────────────────────────────────────────────────────┐
│ Training Loop:                                              │
│                                                             │
│   video: [B, 3, T, H, W]  ← Raw video frames              │
│     ↓                                                       │
│   Encoder → mean, logvar  [B, 12, t, h, w]                │
│     ↓                                                       │
│   Sample z ~ N(mean, var)                                  │
│     ↓                                                       │
│   Decoder → video_recon   [B, 3, T, H, W]                 │
│                                                             │
│   Loss = MSE(video, video_recon) + KL(z || N(0, I))       │
│                                                             │
│ Trainable: Encoder + Decoder (362M params)                 │
│ Frozen: Nothing (this is first stage!)                     │
│ Supervised: NO - completely unsupervised!                  │
│ Data needed: Raw videos only                               │
│ Duration: ~Weeks on multi-GPU cluster                      │
└─────────────────────────────────────────────────────────────┘

After Stage 1:
  ✓ VAE encoder and decoder are fully trained
  ✗ VAE will NEVER be updated again
  ✓ Can now compress any video to 128× smaller latent space

┌────────────────────────────────────────────────────────────────┐
│              STAGE 2: DiT Pre-Training                         │
│              (Diffusion Transformer from scratch)              │
└────────────────────────────────────────────────────────────────┘

Input: Video-caption pairs (supervised learning!)
Goal: Learn text-to-video generation

┌─────────────────────────────────────────────────────────────┐
│ Training Loop:                                              │
│                                                             │
│   video, caption = next(dataloader)                        │
│                                                             │
│   # Encode video (frozen VAE!)                             │
│   with torch.no_grad():                                    │
│       z = vae_encoder(video)  ← Frozen!                   │
│       z = vae_latents_to_dit_latents(z)                    │
│                                                             │
│   # Encode caption (frozen T5!)                            │
│   with torch.no_grad():                                    │
│       y_feat, y_mask = t5_encode(caption)  ← Frozen!      │
│                                                             │
│   # Add noise and predict                                  │
│   eps = torch.randn_like(z)                                │
│   sigma = torch.rand(B)                                    │
│   z_noisy = (1 - sigma) * z + sigma * eps                 │
│   u_target = z - eps                                       │
│                                                             │
│   u_pred = dit(z_noisy, sigma, y_feat, y_mask)            │
│   loss = MSE(u_pred, u_target)                             │
│                                                             │
│ Trainable: DiT only (10B params)                           │
│ Frozen: VAE encoder/decoder (362M), T5-XXL (4.7B)          │
│ Supervised: YES - requires video-caption pairs             │
│ Data needed: Large-scale video-caption dataset            │
│ Duration: ~Months on massive GPU cluster                   │
└─────────────────────────────────────────────────────────────┘

After Stage 2:
  ✓ DiT is fully pre-trained on general text-to-video
  ✓ Can generate diverse videos from any prompt
  ✗ DiT backbone will be frozen for LoRA fine-tuning
  ✓ This is the "Mochi 1 Preview" base model

┌────────────────────────────────────────────────────────────────┐
│              STAGE 3: LoRA Fine-Tuning                         │
│              (Adapt to specific style/content)                 │
└────────────────────────────────────────────────────────────────┘

Input: Small dataset of target videos with captions
Goal: Adapt model to specific style without full retraining

┌─────────────────────────────────────────────────────────────┐
│ Training Loop:                                              │
│                                                             │
│   video, caption = next(dataloader)                        │
│                                                             │
│   # Same as Stage 2, but:                                  │
│   # - VAE frozen (encode once, cache to disk)              │
│   # - T5 frozen (encode once, cache to disk)               │
│   # - DiT backbone frozen (10B params)                     │
│   # - Only LoRA adapters trainable (~96M params)           │
│                                                             │
│   u_pred = dit_with_lora(z_noisy, sigma, y_feat, y_mask)  │
│   loss = MSE(u_pred, u_target)                             │
│                                                             │
│ Trainable: LoRA adapters only (~96M params, <1%)           │
│ Frozen: DiT backbone, VAE, T5 (everything else!)           │
│ Supervised: YES - requires video-caption pairs             │
│ Data needed: Small dataset (10-1000s of videos)            │
│ Duration: ~Hours to days on single H100                    │
└─────────────────────────────────────────────────────────────┘

After Stage 3:
  ✓ LoRA weights adapt base model to specific domain
  ✓ Can merge LoRA into base or keep separate
  ✓ Multiple LoRAs can be trained for different styles
```

### What's Trained When?

```
┌────────────────────────────────────────────────────────────────┐
│              Component Training Timeline                       │
└────────────────────────────────────────────────────────────────┘

                     Stage 1    Stage 2    Stage 3
Component            (VAE)      (DiT)      (LoRA)
─────────────────────────────────────────────────────────
VAE Encoder          ✓ TRAIN    ✗ frozen   ✗ frozen
VAE Decoder          ✓ TRAIN    ✗ frozen   ✗ frozen
T5-XXL               ✗ frozen   ✗ frozen   ✗ frozen
DiT Backbone         N/A        ✓ TRAIN    ✗ frozen
LoRA Adapters        N/A        N/A        ✓ TRAIN

Supervision:         NO         YES        YES
Captions needed:     NO         YES        YES
Typical data size:   Millions   Millions   100s-1000s
Typical duration:    Weeks      Months     Hours-Days
Typical hardware:    Multi-GPU  Huge       Single GPU
                     cluster    cluster    (H100)
```

### Why This Three-Stage Approach?

```
┌────────────────────────────────────────────────────────────────┐
│                        Design Rationale                        │
└────────────────────────────────────────────────────────────────┘

Stage 1 (VAE) - Unsupervised:
  ✓ Video compression doesn't need labels
  ✓ Can use any video data (YouTube, etc.)
  ✓ Much larger dataset available (no captions needed)
  ✓ Learns generic video representation
  ✗ Never needs retraining

Stage 2 (DiT) - Supervised:
  ✓ Text-to-video requires captions
  ✓ Works in compressed latent space (128× faster!)
  ✓ Can focus compute on diffusion modeling
  ✓ Benefits from frozen VAE (stable latents)
  ✗ Expensive, but only done once

Stage 3 (LoRA) - Efficient adaptation:
  ✓ Users can fine-tune on consumer hardware
  ✓ Small datasets sufficient (overfitting controlled by frozen base)
  ✓ Fast iteration (hours not months)
  ✓ Multiple LoRAs for different styles
  ✗ Limited to style/content adaptation (not new capabilities)

This is the same approach used by Stable Diffusion, Flux, etc.
The key insight: separate compression (VAE) from generation (DiT)!
```

### Diffusion Training

```
┌────────────────────────────────────────────────────────────────┐
│                  Diffusion Training (LoRA)                     │
└────────────────────────────────────────────────────────────────┘

for step in range(num_steps):
    # Sample batch
    video, caption = next(dataloader)

    # Encode video to latents (frozen VAE encoder)
    with torch.no_grad():
        latent_dist = encoder(video)  # [B, 12, t, h, w] mean + logvar
        z = latent_dist.sample()
        z = vae_latents_to_dit_latents(z)  # normalize

    # Encode caption (frozen T5)
    with torch.no_grad():
        y_feat, y_mask = t5_encode(caption)  # [B, 256, 4096]

    # Sample noise and timestep
    eps = torch.randn_like(z)
    sigma = torch.rand(B)  # uniform [0, 1]

    # Add noise: z_t = (1 - σ) * z + σ * ε
    z_noisy = (1 - sigma) * z + sigma * eps

    # Predict velocity: v = z - ε
    u_target = vae_latents_to_dit_latents(z - eps)

    # Forward pass
    u_pred = dit(z_noisy, sigma, y_feat, y_mask)

    # MSE loss
    loss = F.mse_loss(u_pred, u_target)
    loss.backward()

    optimizer.step()
    scheduler.step()
```

### Inference (Sampling)

```
┌────────────────────────────────────────────────────────────────┐
│              Euler Sampling (64 steps default)                 │
└────────────────────────────────────────────────────────────────┘

# Initialize
z = torch.randn([B, 12, t, h, w])  # Pure noise

# Encode prompt
y_feat, y_mask = t5_encode(prompt)

# Sigma schedule (linear-quadratic)
sigma_schedule = linear_quadratic_schedule(num_steps=64, threshold=0.025)
# [1.0, 0.985, 0.970, ..., 0.025, 0.0]

# CFG schedule
cfg_schedule = [6.0] * 64  # Classifier-free guidance scale

for i in range(64):
    sigma_curr = sigma_schedule[i]
    sigma_next = sigma_schedule[i + 1]
    dsigma = sigma_curr - sigma_next

    # Conditional prediction
    v_cond = dit(z, sigma_curr, y_feat, y_mask)

    # Unconditional prediction (empty prompt)
    v_uncond = dit(z, sigma_curr, empty_feat, empty_mask)

    # Classifier-free guidance
    cfg_scale = cfg_schedule[i]
    v = v_uncond + cfg_scale * (v_cond - v_uncond)

    # Euler step
    z = z + dsigma * v

# Denormalize and decode
z = dit_latents_to_vae_latents(z)
video = decoder(z)  # [B, 3, T, H, W]
```

---

## LoRA Integration

LoRA (Low-Rank Adaptation) is applied to all QKV and output projections in the attention layers.

```
┌────────────────────────────────────────────────────────────────┐
│                    LoRA Architecture                           │
└────────────────────────────────────────────────────────────────┘

Standard Linear: y = W @ x + b

LoRA Linear: y = (W + ΔW) @ x + b
  where ΔW = (B @ A) * (alpha / r)

  W: [out_dim, in_dim]  frozen pretrained weights
  A: [r, in_dim]        trainable down-projection
  B: [out_dim, r]       trainable up-projection (init to 0)
  alpha: scaling factor (typically = r)
  r: rank (typically 8-64)

┌─ LoRA in AsymmetricAttention ──────────────────────────────┐
│                                                             │
│ qkv_x: LoraLinear(3072 → 9216, r=32, α=32)                │
│   W: [9216, 3072]  ← frozen                                │
│   A: [32, 3072]    ← trainable (~98K params)               │
│   B: [9216, 32]    ← trainable (~295K params)              │
│   Total: 393K trainable vs 28M frozen                      │
│                                                             │
│ qkv_y: LoraLinear(1536 → 9216, r=32, α=32)                │
│   Similar structure                                         │
│                                                             │
│ proj_x: LoraLinear(3072 → 3072, r=32, α=32)               │
│ proj_y: LoraLinear(3072 → 1536, r=32, α=32)               │
│                                                             │
│ Per layer: ~2M trainable LoRA params                        │
│ 48 layers: ~96M trainable params                            │
│ vs 10B frozen params (0.96% trainable)                      │
└─────────────────────────────────────────────────────────────┘

Forward pass:
  def forward(x):
      # Frozen path
      out = F.linear(x, self.weight, self.bias)

      # LoRA path
      if self.r > 0:
          lora_out = (x @ self.lora_A.T) @ self.lora_B.T
          out = out + self.lora_dropout(lora_out) * (self.lora_alpha / self.r)

      return out
```

### LoRA Fine-tuning Configuration

```yaml
# demos/fine_tuner/configs/lora.yaml

model:
  type: lora
  kwargs:
    # Which layers to apply LoRA
    qkv_proj_lora_rank: 32      # Rank for Q, K, V projections
    qkv_proj_lora_alpha: 32
    out_proj_lora_rank: 32      # Rank for output projections
    out_proj_lora_alpha: 32

training:
  num_steps: 2000
  learning_rate: 1e-4
  grad_clip: 1.0

  # Gradient checkpointing for memory
  num_qkv_checkpoint: 48        # Checkpoint all QKV computations
  num_ff_checkpoint: 48         # Checkpoint all feedforward
  num_post_attn_checkpoint: 48  # Checkpoint post-attention

  caption_dropout: 0.1          # Unconditional training 10% of time
```

---

## Common Misconceptions

Based on common questions, here are clarifications about confusing aspects of Mochi's architecture:

### Misconception 1: "Why two different text dimensions (3072 and 1536)?"

```
❌ WRONG: "Text encoding is inconsistent"

✓ CORRECT: Text plays TWO separate roles:
  1. Global conditioning (3072-dim):
     - Single vector per sample
     - Modulates all LayerNorms
     - MUST be 3072 to match visual for modulation
     - Combined with timestep embedding

  2. Dense token features (1536-dim):
     - 256 separate token vectors
     - Participate in joint attention
     - Can be smaller for efficiency (text is simpler)
     - Upprojects to 3072 only during attention

These are NOT two different encodings of the same thing!
They serve completely different purposes.
```

### Misconception 2: "Why is it 4× more parameters when dimensions are only 2×?"

```
❌ WRONG: "2× dimension = 2× parameters"

✓ CORRECT: Parameter scaling is more complex:
  - MLP parameters scale QUADRATICALLY with dimension
  - Visual MLP: (3072)² × mlp_ratio = (3072)² × 8
  - Text MLP: (1536)² × mlp_ratio = (1536)² × 4
  - Ratio: [(3072/1536)² × (8/4)] = 4 × 2 = 8× for MLPs
  - Attention: Only 2× (both project to same space)
  - Weighted average: ~2.65× total, but MLP-dominant

The "4×" or "~5×" refers to effective capacity difference,
not a simple parameter ratio!
```

### Misconception 3: "Does the VAE get updated during LoRA training?"

```
❌ WRONG: "Everything trains together"

✓ CORRECT: Three-stage training:
  Stage 1: Train VAE alone (unsupervised, no captions)
           → Frozen FOREVER after this stage

  Stage 2: Train DiT with frozen VAE
           → DiT backbone frozen after this stage

  Stage 3: Train LoRA with everything else frozen
           → Only ~96M LoRA params trainable

The VAE is NEVER updated after Stage 1.
During LoRA training, VAE is only used to:
  - Preprocessing: Encode videos to latents (done once, cached)
  - Validation: Decode samples to check quality

No gradients flow to VAE, ever!
```

### Misconception 4: "Is the VAE encoder or decoder bigger?"

```
❌ WRONG: "They're the same size (symmetric)"

✓ CORRECT: Decoder is 2-2.5× larger:
  - Encoder: ~100-120M params (64 base channels, max 384)
  - Decoder: ~240-260M params (128 base channels, starts at 768)

Why? Decoding (generation) is harder than encoding (compression):
  - Encoder: Compress while preserving information
  - Decoder: "Hallucinate" fine details from compressed latents
  - Similar to image VAEs, video VAEs need bigger decoders

Total VAE: ~362M params (not 2×181M!)
```

### Misconception 5: "Why upproject text instead of downproject visual?"

```
❌ WRONG: "Should unify everything to same dimension"

✓ CORRECT: Asymmetric approach is most efficient:

  Option A - Downproject visual (bad):
    - 50,000 visual tokens × 3072 → 1536
    - Cost: 50K × 3072 × 1536 = 236 GFLOPs
    - Visual loses capacity (quality degradation)

  Option B - Keep text at 3072 (wasteful):
    - Text MLPs: 48 × 75M = 3.6B wasted params
    - Text doesn't need that capacity

  Option C - Upproject text (optimal!):
    - Only 256 text tokens × 1536 → 3072
    - Cost: 256 × 1536 × 9216 = 3.6 GFLOPs
    - 65× cheaper than downprojecting visual!
    - Text MLPs save 2.7B params
    - Visual keeps full capacity

Non-square projections = best of both worlds!
```

### Misconception 6: "T5 is trained along with the model"

```
❌ WRONG: "T5 is part of the training"

✓ CORRECT: T5-XXL is ALWAYS frozen:
  - Pre-trained by Google (not by Genmo)
  - Frozen in Stage 2 (DiT training)
  - Frozen in Stage 3 (LoRA training)
  - Only used for encoding text to features

In LoRA training, T5 encoding is done ONCE during preprocessing
and cached to disk. The cached embeddings are loaded during training.
No forward pass through T5 happens during training loop!

Components that are NEVER trained by Genmo:
  - T5-XXL (pre-trained by Google)
```

### Misconception 7: "LoRA adds new layers to the model"

```
❌ WRONG: "LoRA inserts new adapter layers"

✓ CORRECT: LoRA modifies existing linear layers:
  - Standard: y = W @ x
  - With LoRA: y = (W + ΔW) @ x, where ΔW = (B @ A) * scale

  W: [out, in] frozen pretrained weight
  A: [r, in] trainable low-rank down-projection
  B: [out, r] trainable low-rank up-projection

No new layers! Just adds low-rank "delta" to existing layers.
The architecture stays EXACTLY the same.

Only affects:
  - QKV projections in attention
  - Output projections in attention

MLPs stay completely frozen (no LoRA on MLPs by default).
```

### Misconception 8: "The model needs 10B × 4 bytes = 40GB minimum"

```
❌ WRONG: "All params must be in full precision"

✓ CORRECT: Memory optimization techniques:
  - DiT backbone: bf16 (2 bytes/param) = 20GB
  - LoRA params: fp32 (4 bytes/param) = 384MB
  - VAE: bf16 = 724MB
  - T5: bf16 = 9.4GB

  With CPU offloading:
    - Keep only active component on GPU
    - T5: GPU during encoding, CPU otherwise
    - VAE: GPU during decode, CPU otherwise
    - DiT: Always on GPU (needed every step)

  With gradient checkpointing:
    - Recompute activations instead of storing
    - ~40GB peak with checkpointing
    - ~70GB+ without (OOM on 80GB GPU)

H100 80GB is sufficient for LoRA training with these optimizations!
```

### Quick Reference: What's Frozen vs Trainable

```
┌────────────────────────────────────────────────────────────────┐
│                  Frozen vs Trainable Summary                   │
└────────────────────────────────────────────────────────────────┘

Component              Params    Stage 1  Stage 2  Stage 3  Used In
                                 (VAE)    (DiT)    (LoRA)   Runtime
──────────────────────────────────────────────────────────────────
VAE Encoder            ~120M     TRAIN    frozen   frozen   Preproc
VAE Decoder            ~240M     TRAIN    frozen   frozen   Decode
T5-XXL                 4.7B      frozen   frozen   frozen   Preproc
DiT Backbone           10B       N/A      TRAIN    frozen   Always
LoRA Adapters          ~96M      N/A      N/A      TRAIN    Always
──────────────────────────────────────────────────────────────────

Preproc = Preprocessing only (cached to disk)
Always = Every forward pass during training/inference
```

---

## Memory and Compute Analysis

### Memory Breakdown (Single H100, 37 frames, 480p)

```
┌────────────────────────────────────────────────────────────────┐
│              Memory Usage (Training with LoRA)                 │
└────────────────────────────────────────────────────────────────┘

Model Parameters:
  - DiT frozen (bf16):      10B × 2 bytes = 20 GB
  - DiT LoRA (fp32):        96M × 4 bytes = 384 MB
  - VAE encoder (bf16):     181M × 2 bytes = 362 MB
  - VAE decoder (bf16):     181M × 2 bytes = 362 MB
  - T5-XXL (bf16):          4.7B × 2 bytes = 9.4 GB
  Total: ~30.5 GB

Activations (batch_size=1, 37 frames, 480×848):
  - Input latents:          1×12×7×60×106 × 2 = 0.1 GB
  - Visual tokens (N):      1×49,290×3072 × 2 = 0.3 GB
  - Text tokens (L):        1×256×1536 × 2 = 0.8 MB
  - Attention (per layer):  ~0.5 GB (Q, K, V, output)
  - MLP (per layer):        ~0.3 GB
  - With checkpointing:     ~8 GB for all 48 layers
  - Gradients (LoRA only):  384 MB × 2 = 768 MB
  Total activations: ~9 GB

Optimizer State (AdamW):
  - LoRA params (fp32):     384 MB
  - Momentum:               384 MB
  - Variance:               384 MB
  Total: ~1.2 GB

Peak Memory: ~41 GB (with gradient checkpointing)
Without checkpointing: ~70+ GB (OOM on H100 80GB)
```

### Compute Analysis (Single Forward Pass)

```
┌────────────────────────────────────────────────────────────────┐
│                  FLOPs Breakdown (480p, 37 frames)             │
└────────────────────────────────────────────────────────────────┘

Visual tokens: N = 49,290
Text tokens: L = 256

Per AsymmetricJointBlock:
  ┌─ Attention ─────────────────────────────────────────────┐
  │ QKV projections:                                         │
  │   Visual: 3×(N × 3072 × 3072) = 1.37 TFLOPs            │
  │   Text:   3×(L × 1536 × 3072) = 3.6 GFLOPs             │
  │ Attention computation:                                   │
  │   QK^T: (N+L) × (N+L) × 3072 = 7.5 TFLOPs              │
  │   Softmax: ~0.5 TFLOPs                                  │
  │   Attn @ V: (N+L) × (N+L) × 3072 = 7.5 TFLOPs          │
  │ Output projection:                                       │
  │   Visual: N × 3072 × 3072 = 456 GFLOPs                 │
  │   Text: L × 3072 × 1536 = 1.2 GFLOPs                   │
  │ Subtotal: ~17 TFLOPs                                    │
  └──────────────────────────────────────────────────────────┘

  ┌─ Feedforward ───────────────────────────────────────────┐
  │ Visual MLP:                                              │
  │   Up: N × 3072 × 12288 = 1.8 TFLOPs                    │
  │   Down: N × 12288 × 3072 = 1.8 TFLOPs                  │
  │ Text MLP:                                                │
  │   Up: L × 1536 × 6144 = 2.4 GFLOPs                     │
  │   Down: L × 6144 × 1536 = 2.4 GFLOPs                   │
  │ Subtotal: ~3.6 TFLOPs                                   │
  └──────────────────────────────────────────────────────────┘

Per block total: ~20.6 TFLOPs
48 blocks: ~990 TFLOPs

Other components:
  - Patch embedding: ~5 GFLOPs
  - T5 encoding: ~100 TFLOPs (done once, cached)
  - Final layer: ~0.5 GFLOPs

Single forward pass: ~1 PFLOPs (1000 TFLOPs)

H100 throughput: ~1000 TFLOPs/s (bf16)
Theoretical time: ~1 second/step
Actual time: ~1.67 seconds/step (memory bandwidth bound)

64 sampling steps: ~2 minutes for 37 frames
```

---

## Context Parallel (Multi-GPU)

For multi-GPU inference, Mochi splits the temporal dimension across GPUs.

```
┌────────────────────────────────────────────────────────────────┐
│              Context Parallel (2 GPUs example)                 │
└────────────────────────────────────────────────────────────────┘

Video: [B, C, T=31, H, W]

GPU 0: [B, C, T=16, H, W]  (frames 0-15)
GPU 1: [B, C, T=15, H, W]  (frames 15-30)
        ↑ overlap at frame 15

┌─ Visual Token Split ───────────────────────────────────────┐
│ After patch embedding: N = 31 × 30 × 53 = 49,290          │
│                                                             │
│ GPU 0: M = 24,645 tokens  (frames 0-15)                    │
│ GPU 1: M = 24,645 tokens  (frames 15-30)                   │
└─────────────────────────────────────────────────────────────┘

┌─ Attention Computation ────────────────────────────────────┐
│ Each GPU computes attention for its local M tokens         │
│                                                             │
│ Head split:                                                 │
│   GPU 0: heads 0-11   (local_heads = 12)                   │
│   GPU 1: heads 12-23  (local_heads = 12)                   │
│                                                             │
│ Communication via all-to-all:                               │
│   - all_to_all_collect_tokens: gather QKV from all GPUs    │
│   - all_to_all_collect_heads: gather attention outputs     │
└─────────────────────────────────────────────────────────────┘

Advantages:
  - Split model parameters across GPUs (reduce memory per GPU)
  - Split activation memory (enable longer videos)
  - Communication only during attention (not MLP)

Disadvantages:
  - Requires NCCL/FSDP setup
  - Communication overhead (~20% slower than single GPU)
```

---

## Comparison with Other Architectures

```
┌────────────────────────────────────────────────────────────────┐
│        Mochi vs Other Video Diffusion Models                   │
└────────────────────────────────────────────────────────────────┘

                 Mochi-1    Stable Video  CogVideoX   Pika/Runway
                            Diffusion     -5B         (closed)
─────────────────────────────────────────────────────────────────
Parameters       10B        ~1.5B         5B          Unknown
Architecture     AsymmDiT   U-Net         DiT         Unknown
Text Encoder     T5-XXL     CLIP          T5          Unknown
                 (single)   (dual)
VAE Compression  128x       64x           ?           ?
                 (8×8×6)    (8×8×4)
Latent Channels  12         4             16          Unknown
Open Source      ✓          ✓             ✓           ✗
Max Resolution   480p       576p          720p        1080p+
                 (preview)
─────────────────────────────────────────────────────────────────

Key Innovations in Mochi:
  1. Asymmetric architecture (~2.65× params per block, but MLPs are 4× larger)
  2. Single strong text encoder (T5-XXL) vs dual weak encoders
  3. Higher compression VAE (128x vs 64x)
  4. Joint attention with non-square projections
  5. Mixed 3D RoPE for temporal-spatial awareness
```

---

## Implementation Tips

### For Training

1. **Always enable gradient checkpointing**:
   ```yaml
   num_qkv_checkpoint: 48
   num_ff_checkpoint: 48
   num_post_attn_checkpoint: 48
   ```
   Reduces memory by ~40% at cost of ~20% slower

2. **Use torch.compile**:
   ```bash
   export COMPILE_DIT=1
   ```
   Speeds up by ~15-20% after warmup

3. **Batch size = 1**: Model is memory-intensive, use gradient accumulation if needed

4. **Caption quality matters**: Detailed captions improve results significantly

### For Inference

1. **CPU offloading** for <60GB VRAM:
   ```python
   pipeline = MochiSingleGPUPipeline(cpu_offload=True, ...)
   ```

2. **Tiled decoding** for long videos:
   ```python
   pipeline = MochiSingleGPUPipeline(
       decode_type="tiled_spatial",
       decode_args={"num_tiles_w": 4, "num_tiles_h": 2, "overlap": 8}
   )
   ```

3. **Adjust CFG scale** for quality vs creativity:
   - CFG 4-5: More creative, diverse
   - CFG 6-7: Balanced (default)
   - CFG 8+: More faithful to prompt, less diverse

4. **Sigma schedule tuning**:
   ```python
   # Default: linear_quadratic_schedule(64, 0.025)
   # For sharper results, increase threshold:
   sigma_schedule = linear_quadratic_schedule(64, 0.035)
   ```

---

## References

- Paper: [Mochi 1 Blog Post](https://www.genmo.ai/blog)
- Code: [github.com/genmoai/models](https://github.com/genmoai/models)
- Weights: [Hugging Face](https://huggingface.co/genmo/mochi-1-preview)
- Flash Attention: [Dao et al., 2022](https://arxiv.org/abs/2205.14135)
- LoRA: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- RoPE: [Su et al., 2021](https://arxiv.org/abs/2104.09864)

---

*Last updated: 2024*