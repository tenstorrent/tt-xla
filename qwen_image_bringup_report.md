# Bringup Report: Qwen-Image

## Model Overview


| Property              | Value                                                                            |
| --------------------- | -------------------------------------------------------------------------------- |
| **Model**             | Qwen-Image                                                                       |
| **HuggingFace**       | [https://huggingface.co/Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image) |
| **Type**              | Diffusion-based image generation (MMDiT)                                         |
| **Total Parameters**  | ~27B (20B transformer + 7B text encoder + VAE)                                   |
| **Total Weight Size** | 57.7 GB (BF16)                                                                   |
| **License**           | Apache 2.0                                                                       |
| **Pipeline**          | `QwenImagePipeline` (diffusers)                                                  |


### Architecture

Qwen-Image is a flow-matching diffusion model built on a **Multimodal Diffusion Transformer (MMDiT)** architecture with a dual-stream design. It uses the full Qwen2.5-VL-7B model as its text encoder and an AutoencoderKL VAE for latent space encoding/decoding.

**Inference pipeline flow:**

1. Text Encoder (Qwen2.5-VL-7B) produces text embeddings → offloadable after completion
2. Transformer (MMDiT) runs iterative denoising (default 50 steps) → memory-dominant phase
3. VAE Decoder converts latents to pixel space → lightweight final step

### TT Hardware Reference

> **Correction:** p300 = 2 × p150 (two 32 GB chips), NOT a single 64 GB device. Consequently, ge (2 × p300) = 4 × p150 = same device topology as bhqb.


| Architecture | Composition         | Devices | DRAM/device | Total DRAM |
| ------------ | ------------------- | ------- | ----------- | ---------- |
| n150         | single chip         | 1       | 12 GB       | 12 GB      |
| p100         | single chip         | 1       | 28 GB       | 28 GB      |
| p150         | single chip         | 1       | 32 GB       | 32 GB      |
| p300         | 2 × p150            | 2       | 32 GB       | 64 GB      |
| n300         | 2 × n150            | 2       | 12 GB       | 24 GB      |
| whqb         | 4 × n300            | 8       | 12 GB       | 96 GB      |
| bhqb         | 4 × p150            | 4       | 32 GB       | 128 GB     |
| ge           | 2 × p300 = 4 × p150 | 4       | 32 GB       | 128 GB     |
| wh galaxy    | 32 × n150           | 32      | 12 GB       | 384 GB     |
| bh galaxy    | 32 × p150           | 32      | 32 GB       | 1024 GB    |


---

## Model Configurations

```yaml
model_name: "Qwen-Image"
huggingface_repo: "https://huggingface.co/Qwen/Qwen-Image"

configurations:
  - name: "text_to_image"
    total_memory: "57.7 GB"
    precision: "bfloat16"
    transformer_memory: "40.9 GB"
    text_encoder_memory: "16.6 GB"
    vae_decoder_memory: "0.254 GB"
    peak_memory_sequential: "40.9 GB"
    peak_memory_all_loaded: "57.7 GB"
    denoising_steps: 50

  - name: "image_to_image"
    total_memory: "57.7 GB"
    precision: "bfloat16"
    transformer_memory: "40.9 GB"
    text_encoder_memory: "16.6 GB"
    vae_encoder_memory: "0.254 GB"
    vae_decoder_memory: "0.254 GB"
    peak_memory_sequential: "40.9 GB"
    peak_memory_all_loaded: "57.7 GB"
    denoising_steps: 50
```

---

## Component Shardability Analysis

### Transformer (MMDiT) — 40.9 GB


| Layer Type                       | Shardable?  | Notes                                              |
| -------------------------------- | ----------- | -------------------------------------------------- |
| Attention Q/K/V projections      | Yes         | 24 heads × 128 dim, divisible by 1,2,3,4,6,8,12,24 |
| Attention output projections     | Yes         | Shard across head dimension                        |
| MLP/FFN layers                   | Yes         | Shard across hidden dimension                      |
| RMSNorm / AdaLN                  | No          | Must replicate (~negligible size)                  |
| Timestep/position embeddings     | No          | Must replicate (~negligible size)                  |
| **Estimated shardable fraction** | **~95-98%** | 60 identical layers = excellent tensor parallelism |


Shardable weights: ~38.9-40.1 GB. Replicated weights: ~0.8-2.0 GB.

### Text Encoder (Qwen2.5-VL-7B) — 16.6 GB


| Layer Type                       | Shardable?  | Notes                                       |
| -------------------------------- | ----------- | ------------------------------------------- |
| Attention layers                 | Yes         | 28 heads, GQA with 4 KV heads               |
| MLP/FFN layers                   | Yes         | Shard across intermediate dim (18,944)      |
| Embeddings                       | No          | Vocab embeddings replicated                 |
| Layer norms                      | No          | Must replicate                              |
| **Estimated shardable fraction** | **~90-95%** | Standard transformer, well-studied sharding |


Shardable weights: ~14.9-15.8 GB. Replicated weights: ~0.8-1.7 GB.

### VAE — 0.254 GB

Convolutional architecture. Not practically shardable across devices. Runs on a single device. Negligible size — not a bottleneck.

---

## Activation Memory Estimation

Activations are transient memory consumed during forward passes, on top of model weights.


| Component                  | Estimated Activation Memory | Notes                                                                   |
| -------------------------- | --------------------------- | ----------------------------------------------------------------------- |
| Transformer (per step)     | ~8-16 GB                    | 60 layers × latent size (128×128×16 for 1024px). Depends on resolution. |
| Text Encoder (single pass) | ~2-4 GB                     | 7B model at typical sequence lengths (≤512 tokens)                      |
| VAE Decoder (single pass)  | ~0.5-1 GB                   | Single pass, small model                                                |
| **Conservative total**     | **~10-20 GB**               | Peak during transformer denoising phase                                 |


For multi-device setups, activations are partially distributed (attention activations shard with heads), but some intermediate tensors are replicated during all-reduce operations.

---

## Valid Workloads on Device

> Note: Both configurations (text_to_image and image_to_image) have identical memory profiles since the VAE encoder/decoder share the same 0.254 GB model. All analysis below applies to both configurations.

---

### n150 — 1 device, 12 GB DRAM

```
arch_name: n150
total_memory_weights: 57.7 GB (all loaded) / 40.9 GB (sequential)
theoretical_min_weights_memory_per_device: 57.7 GB (single device, no sharding possible)
```

**comment:** NOT FEASIBLE. Even the transformer alone (40.9 GB) exceeds the 12 GB device DRAM by 3.4×. No viable path.

---

### p100 — 1 device, 28 GB DRAM

```
arch_name: p100
total_memory_weights: 57.7 GB (all loaded) / 40.9 GB (sequential)
theoretical_min_weights_memory_per_device: 57.7 GB (single device)
```

**comment:** NOT FEASIBLE. The transformer alone (40.9 GB) exceeds the 28 GB device DRAM by 1.46×. Even aggressive quantization to INT8 (~20.5 GB) would leave minimal room for activations.

---

### p150 — 1 device, 32 GB DRAM

```
arch_name: p150
total_memory_weights: 57.7 GB (all loaded) / 40.9 GB (sequential)
theoretical_min_weights_memory_per_device: 57.7 GB (single device)
```

**comment:** NOT FEASIBLE. The transformer alone (40.9 GB) exceeds the 32 GB device DRAM by 1.28×. No single-chip option can hold this model's transformer in BF16.

---

### p300 — 2 × p150, 2 devices × 32 GB DRAM

```
arch_name: p300
total_memory_weights: 57.7 GB
theoretical_min_weights_memory_per_device: 28.85 GB (if 100% shardable)
```

**Realistic calculation (transformer only, sequential):**

- Shardable: ~38.9 GB / 2 = 19.45 GB per device
- Replicated: ~2.0 GB per device
- **Per device: ~21.5 GB** → 32 GB - 21.5 GB = **10.5 GB remaining** for activations

**All components sharded across 2 devices:**

- Transformer: ~21.5 GB/dev + Text Encoder: ~9.0 GB/dev + VAE: ~0.254 GB/dev (replicated)
- **Per device: ~30.8 GB** → **1.2 GB remaining** — too tight for activations

**Tensor parallelism notes:**

- 24 attention heads ÷ 2 devices = 12 heads/device ✓ (clean division)
- 60 layers ÷ 2 devices = 30 layers/device ✓ (clean division)

**comment:** FEASIBLE with sequential offloading — this is the **minimum viable multi-device option**. With sequential offloading (only the transformer loaded during denoising), each device has ~10.5 GB for activations. This should be sufficient for standard resolutions (512×512, 1024×1024), though high-res generation (2048×2048) may be tight. All-loaded mode is NOT feasible (only 1.2 GB remaining per device). The 24 attention heads and 60 layers both divide evenly by 2, making tensor parallelism clean.

---

### n300 — 2 × n150, 2 devices × 12 GB DRAM

```
arch_name: n300
total_memory_weights: 57.7 GB
theoretical_min_weights_memory_per_device: 28.85 GB (if 100% shardable)
```

**Realistic calculation (transformer only, sequential):**

- Shardable: ~38.9 GB / 2 = 19.45 GB per device
- Replicated: ~2.0 GB per device
- **Per device: ~21.5 GB** → exceeds 12 GB DRAM

**comment:** NOT FEASIBLE. Even sharding just the transformer across 2 devices requires ~21.5 GB per device, which is 1.79× the available 12 GB DRAM. Same device count as p300 but with 12 GB chips instead of 32 GB — far too small.

---

### whqb (Wormhole QuietBox) — 4 × n300 = 8 devices × 12 GB DRAM

```
arch_name: whqb
total_memory_weights: 57.7 GB
theoretical_min_weights_memory_per_device: 7.21 GB (if 100% shardable)
```

**Realistic calculation (transformer only, sequential):**

- Shardable: ~38.9 GB / 8 = 4.86 GB per device
- Replicated: ~2.0 GB per device
- **Per device: ~6.9 GB** → 12 GB - 6.9 GB = **5.1 GB remaining** for activations

**All components sharded across 8 devices:**

- Transformer: ~6.9 GB/dev + Text Encoder: ~3.6 GB/dev + VAE: ~0.254 GB/dev (replicated)
- **Per device: ~10.8 GB** → **1.2 GB remaining** — too tight

**Tensor parallelism notes:**

- 24 attention heads ÷ 8 devices = 3 heads/device ✓ (clean division)
- 60 layers ÷ 8 devices = 7.5 layers/device (pipeline parallelism would use 7 or 8 layers per device)

**comment:** FEASIBLE with sequential offloading, but activation headroom is tight at 5.1 GB per device. Suitable for moderate image sizes (512×512). For 1024×1024 generation, activations may exceed the 5.1 GB budget. All-loaded mode is NOT feasible (only 1.2 GB remaining). The 24 heads divide cleanly across 8 devices, making tensor parallelism straightforward. However, communication overhead across 8 wormhole chips is non-trivial.

---

### bhqb (Blackhole QuietBox) — 4 × p150 = 4 devices × 32 GB DRAM

```
arch_name: bhqb
total_memory_weights: 57.7 GB
theoretical_min_weights_memory_per_device: 14.43 GB (if 100% shardable)
```

**Realistic calculation (transformer only, sequential):**

- Shardable: ~38.9 GB / 4 = 9.73 GB per device
- Replicated: ~2.0 GB per device
- **Per device: ~11.7 GB** → 32 GB - 11.7 GB = **20.3 GB remaining** for activations ✓ Excellent

**All components sharded across 4 devices:**

- Transformer: ~11.7 GB/dev + Text Encoder: ~5.4 GB/dev + VAE: ~0.254 GB/dev (replicated)
- **Per device: ~17.4 GB** → **14.6 GB remaining** for activations ✓ Very good

**Tensor parallelism notes:**

- 24 attention heads ÷ 4 devices = 6 heads/device ✓ (clean division)
- 60 layers ÷ 4 devices = 15 layers/device ✓ (clean division)
- 28 text encoder heads ÷ 4 devices = 7 heads/device ✓

**comment:** FEASIBLE — this is the **recommended configuration**. Excellent activation headroom in both sequential (20.3 GB) and all-loaded (14.6 GB) modes. Perfect tensor parallelism alignment: 24 heads and 60 layers both divide evenly by 4. Can comfortably handle 1024×1024 and potentially larger resolutions. All components can remain loaded simultaneously, avoiding offloading overhead during the 50-step denoising loop.

---

### ge (QuietBox2) — 2 × p300 = 4 × p150 = 4 devices × 32 GB DRAM

```
arch_name: ge
total_memory_weights: 57.7 GB
theoretical_min_weights_memory_per_device: 14.43 GB (if 100% shardable)
```

**Analysis: Identical to bhqb.** Since p300 = 2 × p150, ge (2 × p300) = 4 × p150 = same device topology and memory as bhqb.

- Sequential (transformer only): ~11.7 GB/dev → **20.3 GB remaining** ✓
- All loaded: ~17.4 GB/dev → **14.6 GB remaining** ✓
- Tensor parallelism: 24 heads ÷ 4 = 6/dev ✓, 60 layers ÷ 4 = 15/dev ✓

**comment:** FEASIBLE — same as bhqb. The only potential difference is interconnect topology (2 × p300 boards vs. 4 separate p150 boards), which may affect inter-device communication bandwidth. From a memory perspective, the analysis is identical.

---

### wh galaxy (Wormhole Galaxy) — 32 × n150 = 32 devices × 12 GB DRAM

```
arch_name: wh_galaxy
total_memory_weights: 57.7 GB
theoretical_min_weights_memory_per_device: 1.80 GB (if 100% shardable)
```

**Realistic calculation (transformer only, sequential):**

- Shardable: ~38.9 GB / 32 = 1.22 GB per device
- Replicated: ~2.0 GB per device
- **Per device: ~3.2 GB** → 12 GB - 3.2 GB = **8.8 GB remaining** for activations

**Tensor parallelism notes:**

- 24 attention heads ÷ 32 devices → does NOT divide evenly
- Would need to use a subset (e.g., 24 devices for transformer, remaining 8 idle or for other tasks)

**comment:** FEASIBLE but impractical for single inference. The 24 attention heads don't divide evenly by 32, complicating tensor parallelism. Communication overhead across 32 wormhole devices dominates for a single inference pass. Better suited for high-throughput serving using subsets: run multiple 4- or 8-device pipeline instances in parallel (e.g., 4 independent bhqb-equivalent pipelines on 16 devices, or 8 p300-equivalent pipelines on 16 devices).

---

### bh galaxy (Blackhole Galaxy) — 32 × p150 = 32 devices × 32 GB DRAM

```
arch_name: bh_galaxy
total_memory_weights: 57.7 GB
theoretical_min_weights_memory_per_device: 1.80 GB (if 100% shardable)
```

**Realistic calculation (transformer only, sequential):**

- Shardable: ~38.9 GB / 32 = 1.22 GB per device
- Replicated: ~2.0 GB per device
- **Per device: ~3.2 GB** → 32 GB - 3.2 GB = **28.8 GB remaining** for activations

**comment:** FEASIBLE but extreme overkill for single inference. Each device would have 28.8 GB of unused DRAM. Same attention head divisibility issue as wh galaxy (24 heads vs 32 devices). Recommended: partition into 8 independent bhqb-equivalent (4-device) pipeline instances for maximum throughput, serving 8 concurrent image generations.

---

## Summary

### Feasibility Matrix


| Architecture | Devices | DRAM/dev | Sequential Offload   | All Loaded           | Recommended?           |
| ------------ | ------- | -------- | -------------------- | -------------------- | ---------------------- |
| n150         | 1       | 12 GB    | ❌                    | ❌                    | No                     |
| p100         | 1       | 28 GB    | ❌                    | ❌                    | No                     |
| p150         | 1       | 32 GB    | ❌                    | ❌                    | No                     |
| **p300**     | 2       | 32 GB    | ✅ (10.5 GB free/dev) | ❌ (1.2 GB free)      | Yes — minimum viable   |
| n300         | 2       | 12 GB    | ❌                    | ❌                    | No                     |
| whqb         | 8       | 12 GB    | ⚠️ (5.1 GB free/dev) | ❌ (1.2 GB free)      | Marginal               |
| **bhqb**     | 4       | 32 GB    | ✅ (20.3 GB free/dev) | ✅ (14.6 GB free/dev) | **Yes — recommended**  |
| **ge**       | 4       | 32 GB    | ✅ (20.3 GB free/dev) | ✅ (14.6 GB free/dev) | **Yes — same as bhqb** |
| wh galaxy    | 32      | 12 GB    | ✅ (8.8 GB free/dev)  | ⚠️                   | Use 4-8 dev subset     |
| bh galaxy    | 32      | 32 GB    | ✅ (28.8 GB free/dev) | ✅                    | Use 4-dev subsets      |


### Key Insight: No Single-Device Option

The 40.9 GB MMDiT transformer exceeds every single-chip TT device (max is p150 at 32 GB). **Multi-device sharding is required for any deployment of Qwen-Image on TT hardware.**

### Recommended Hardware Tiers

1. **Recommended: bhqb / ge (4 × p150)** — Perfect tensor parallelism alignment (24 heads ÷ 4, 60 layers ÷ 4), ample activation budget (14.6-20.3 GB/dev), supports all-loaded mode eliminating offloading latency during the 50-step denoising loop
2. **Minimum viable: p300 (2 × p150)** — Works with sequential offloading only, ~10.5 GB activation headroom per device, sufficient for standard resolutions
3. **High-throughput: bh galaxy (32 × p150)** — Partition into 8 independent 4-device pipelines for 8× concurrent image generation throughput
4. **Marginal: whqb (8 × n150)** — Feasible but tight activation budget (5.1 GB/dev) limits resolution flexibility

### Key Constraints

- The 40.9 GB MMDiT transformer is the dominant memory consumer and requires multi-device sharding on all TT architectures
- Sequential offloading (text encoder → transformer → VAE) is essential for 2-device configurations (p300)
- 24 attention heads provide clean tensor parallelism for 2, 3, 4, 6, 8, 12, or 24 device configurations
- 60 transformer layers enable clean pipeline parallelism for 2, 3, 4, 5, 6, 10, 12, 15, 20, or 30 device splits
- Activation memory scales with image resolution — higher resolutions need more headroom
- bhqb and ge are functionally equivalent (both 4 × p150); only interconnect topology may differ

