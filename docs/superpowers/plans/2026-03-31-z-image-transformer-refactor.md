# Z-Image Transformer Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the 35K-line codegenerated `z-transformer-only/main.py` into clean, modular TTNN code with LightweightModule classes, HuggingFace weight loading, and looped transformer blocks — maintaining PCC = 1.0000042915.

**Architecture:** Bottom-up module extraction. Build smallest modules first (RMSNorm, Attention, FeedForward), compose into TransformerBlock, then into ZImageTransformerTTNN. Weights load from PyTorch state_dict via `ttnn.from_torch()`. Const-eval functions deduplicated into `consteval.py`.

**Tech Stack:** TTNN (Tenstorrent neural network library), PyTorch, HuggingFace diffusers (`ZImagePipeline`), `LightweightModule` from `models.common.lightweightmodule`.

**Spec:** `docs/superpowers/specs/2026-03-31-z-image-transformer-refactor-design.md`

**Baseline PCC:** 1.0000042915 against `z-image/tensor_dump_transformer/runtime_out0.pt`

**Key constants used throughout:**
- `DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)`
- `COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)`
- `COMPUTE_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)`
- `RMS_EPS = 9.9999997473787516e-06`

**Key model dimensions:**
- `dim=3840, n_heads=30, head_dim=128, ffn_hidden=10240`
- `cap_feat_dim=2560, in_channels=16, adaln_dim=256`
- `n_refiner_layers=2, n_layers=30`
- `image_tokens=3600 (80×45), image_padded=3616, cap_tokens=18, cap_padded=32`
- `total_seq=3648 (3616+32)`

---

### Task 1: Create `utils.py`

**Files:**
- Create: `z-transformer-only/utils.py` (replace existing)

The existing `utils.py` has DeviceGetter and golden workarounds from the codegen runtime. Replace with clean utilities.

- [ ] **Step 1: Write `utils.py`**

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch


def calculate_pcc(x, y):
    """Calculate Pearson Correlation Coefficient between two tensors."""
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    x_flat, y_flat = x.flatten().float(), y.flatten().float()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom
```

- [ ] **Step 2: Commit**

```bash
git add z-transformer-only/utils.py
git commit -m "refactor(z-transformer): replace utils.py with clean PCC utility"
```

---

### Task 2: Create `model_pt.py`

**Files:**
- Create: `z-transformer-only/model_pt.py`

Standalone PyTorch reference model copied from `z-image/transformer.py` and `z-image/model.py`, with HuggingFace loading.

- [ ] **Step 1: Write `model_pt.py`**

Copy the full PyTorch model from `z-image/transformer.py` (all building blocks: `TimestepEmbedder`, `RMSNorm`, `RealRopeEmbedder`, `FeedForward`, `FinalLayer`, `Attention`, `TransformerBlock`, `ZImageTransformer`) and add the model loading/input functions:

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference implementation of Z-Image Transformer."""

import math
import os
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "Tongyi-MAI/Z-Image"
DTYPE = torch.bfloat16
MODEL_CACHE_PATH = "z_image_transformer.pt"

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

# --- Copy ALL classes from z-image/transformer.py ---
# TimestepEmbedder, RMSNorm, RealRopeEmbedder, FeedForward,
# FinalLayer, _apply_rotary_emb, Attention, TransformerBlock,
# ZImageTransformer (complete with _copy_weights, forward, etc.)
# --- End copy ---


class ZImageTransformerPT(nn.Module):
    """Wrapper that loads the Z-Image transformer from HuggingFace."""

    def __init__(self, cache_path=MODEL_CACHE_PATH):
        super().__init__()
        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self._load_from_huggingface()
            if cache_path:
                self._save_to_cache(cache_path)
        self.eval()

    def _load_from_cache(self, path):
        print(f"Loading model from {path}...")
        self.transformer = torch.load(path, weights_only=False)
        print(f"Model loaded from {path}")

    def _load_from_huggingface(self):
        from diffusers import ZImagePipeline
        from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift

        print(f"Loading Z-Image pipeline from {MODEL_ID}...")
        pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)

        # Extract transformer inputs shape
        latents = pipe.prepare_latents(
            batch_size=1,
            num_channels_latents=pipe.transformer.in_channels,
            height=1280,
            width=720,
            dtype=DTYPE,
            device="cpu",
            generator=torch.Generator().manual_seed(42),
        )
        latent_input = latents.to(DTYPE).unsqueeze(2)[0]  # [C, 1, H, W]

        positive_prompt = "A photo of a cat sitting on a windowsill"
        # Encode prompt to get caption length
        tokens = pipe.tokenizer(
            positive_prompt,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        # Run text encoder to get prompt embeddings
        with torch.inference_mode():
            text_encoder_output = pipe.text_encoder(
                tokens.input_ids,
                attention_mask=tokens.attention_mask.bool(),
                output_hidden_states=True,
            )
        cap_feat = text_encoder_output.hidden_states[-2][0]  # [seq_len, 2560]

        self.transformer = ZImageTransformer(
            pipe.transformer,
            cap_len=cap_feat.shape[0],
            image_shape=tuple(latent_input.shape),
        )
        self.transformer.to(DTYPE)

    def _save_to_cache(self, path):
        print(f"Saving model to {path}...")
        torch.save(self.transformer, path)

    def forward(self, latent_input, timestep, cap_feat):
        return self.transformer(latent_input, timestep, cap_feat)

    def state_dict_for_ttnn(self):
        """Return the transformer's state_dict for TTNN weight loading."""
        return self.transformer.state_dict()


def get_input(model_pt):
    """Generate sample inputs matching what the codegen was traced with."""
    from diffusers import ZImagePipeline

    pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
    latents = pipe.prepare_latents(
        batch_size=1,
        num_channels_latents=pipe.transformer.in_channels,
        height=1280,
        width=720,
        dtype=DTYPE,
        device="cpu",
        generator=torch.Generator().manual_seed(42),
    )
    latent_input = latents.to(DTYPE).unsqueeze(2)[0]  # [16, 1, 160, 90]
    timestep = torch.tensor([0.5])

    positive_prompt = "A photo of a cat sitting on a windowsill"
    tokens = pipe.tokenizer(
        positive_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        text_encoder_output = pipe.text_encoder(
            tokens.input_ids,
            attention_mask=tokens.attention_mask.bool(),
            output_hidden_states=True,
        )
    cap_feat = text_encoder_output.hidden_states[-2][0]  # [seq_len, 2560]

    return latent_input, timestep, cap_feat
```

**Important:** The actual file must contain the COMPLETE class definitions copied from `z-image/transformer.py`. The `# --- Copy ALL classes ---` comment indicates where to paste the ~300 lines of PyTorch model code (lines 1-490 of `z-image/transformer.py`).

- [ ] **Step 2: Verify PyTorch model produces correct reference output**

Run a quick test comparing against the stored reference:
```python
# In z-transformer-only/
python3 -c "
import torch
from model_pt import ZImageTransformerPT, get_input
model = ZImageTransformerPT()
latent, timestep, cap_feat = get_input(model)
with torch.inference_mode():
    out = model(latent, timestep, cap_feat)
print(f'Output shape: {out.shape}')
ref = torch.load('../z-image/tensor_dump_transformer/runtime_out0.pt', weights_only=True)
x, y = ref.flatten().float(), out.flatten().float()
vx, vy = x - x.mean(), y - y.mean()
pcc = (vx @ vy) / (vx.norm() * vy.norm())
print(f'PCC vs reference: {pcc:.10f}')
"
```
Expected: PCC > 0.99 (should be ~1.0)

- [ ] **Step 3: Commit**

```bash
git add z-transformer-only/model_pt.py
git commit -m "refactor(z-transformer): add standalone PyTorch reference model"
```

---

### Task 3: Create `consteval.py`

**Files:**
- Create: `z-transformer-only/consteval.py`

From analysis of the 85 const-eval functions, there are 10 unique patterns. Most are weight format conversions. The complex one (const_eval_22) handles RoPE and attention mask precomputation.

- [ ] **Step 1: Write `consteval.py`**

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Const-eval functions for Z-Image Transformer model.

These transform weights at load time (transpose, cast, reshape, broadcast).
Called once during model initialization, results cached for inference.
"""

import ttnn

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def transpose_and_cast_to_f32(tensor, device):
    """Pattern 0: to_device → TILE → permute([1,0]) → FLOAT32.

    Used for weight matrices that need transposing for matmul.
    37 uses across the model.
    """
    x = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(tensor, False)
    x_perm = ttnn.permute(x, [1, 0], memory_config=DRAM, pad_value=0.0)
    ttnn.deallocate(x, False)
    x_cast = ttnn.typecast(x_perm, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(x_perm, False)
    return x_cast


def cast_to_f32(tensor, device):
    """Pattern 1: to_device → TILE → FLOAT32.

    Used for bias vectors and non-transposed weights.
    35 uses across the model.
    """
    x = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(tensor, False)
    x_cast = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(x, False)
    return x_cast


def reshape_to_tile(tensor, device, shape):
    """Pattern 4: to_device → TILE → reshape.

    Used for RoPE frequency buffers and special reshapes.
    2 uses.
    """
    x = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(tensor, False)
    return ttnn.reshape(x, shape, memory_config=DRAM)


def reshape_and_cast_to_f32(tensor, device, shape):
    """Pattern 6: to_device → TILE → reshape → FLOAT32.

    1 use.
    """
    x = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(tensor, False)
    x = ttnn.reshape(x, shape, memory_config=DRAM)
    x_cast = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(x, False)
    return x_cast


def reshape_cast_and_repeat(tensor, device, shape, repeat_shape):
    """Pattern 5: to_device → TILE → reshape → FLOAT32 → repeat.

    Used for bias broadcast. 2 uses.
    """
    x = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(tensor, False)
    x = ttnn.reshape(x, shape, memory_config=DRAM)
    x = ttnn.typecast(x, ttnn.DataType.FLOAT32, memory_config=DRAM)
    return ttnn.repeat(x, ttnn.Shape(repeat_shape))


def reshape_and_repeat(tensor, device, shape, repeat_shape):
    """Pattern 8: to_device → TILE → reshape → repeat.

    1 use.
    """
    x = ttnn.to_device(tensor, device=device, memory_config=DRAM)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(tensor, False)
    x = ttnn.reshape(x, shape, memory_config=DRAM)
    return ttnn.repeat(x, ttnn.Shape(repeat_shape))


def create_scalar_ones(device):
    """Pattern 3/9: Create scalar ones tensor.

    Used for adaLN `1 + scale` computation.
    """
    return ttnn.full(
        shape=ttnn.Shape([1, 1]),
        fill_value=1.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )


def create_attn_mask(mask_tensor, device, seq_len):
    """Pattern 2: Build attention mask from boolean pad mask.

    Creates a float mask: 0 where valid, -inf where padded.
    3 uses (x_attn_mask, cap_attn_mask, unified_attn_mask).
    """
    x = ttnn.to_device(mask_tensor, device=device, memory_config=DRAM)
    x = ttnn.to_layout(x, ttnn.Layout.TILE, None, memory_config=None)

    neg_inf = ttnn.full(
        shape=x.shape,
        fill_value=float("-inf"),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )
    zeros = ttnn.full(
        shape=x.shape,
        fill_value=0.0,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
        memory_config=DRAM,
    )

    mask = ttnn.reshape(x, [1, 1, 1, seq_len], memory_config=DRAM)
    neg_inf = ttnn.reshape(neg_inf, [1, 1, 1, seq_len], memory_config=DRAM)
    zeros = ttnn.reshape(zeros, [1, 1, 1, seq_len], memory_config=DRAM)

    result = ttnn.where(mask, neg_inf, zeros, memory_config=DRAM)
    return ttnn.repeat(result, ttnn.Shape([1, 1, seq_len, 1]))


def precompute_rope_and_pos(
    cap_pos_ids, sin_2, sin_1, sin_0, cos_2, cos_1, cos_0, image_pos_ids, device
):
    """Pattern 7: Complex RoPE precomputation (const_eval_22).

    Computes positional embeddings by looking up precomputed sin/cos tables
    for 3 axes (temporal, height, width), then concatenates.

    This replaces main_const_eval_22 which was 150+ ops.
    Must be extracted from the exact codegen op sequence (lines 658-1776).
    """
    # This function must replicate the EXACT op sequence from main_const_eval_22.
    # It takes 8 inputs and returns 6 outputs:
    #   - image_freqs_cis (for image tokens)
    #   - cap_freqs_cis (for caption tokens)
    #   - additional position-related tensors
    #
    # The implementation should be extracted directly from the codegen
    # lines 658-1776 of main.py, with variable names cleaned up.
    # The exact sequence involves:
    #   1. Gather cos/sin values using position IDs for each axis
    #   2. Concatenate across axes: [cos_0, cos_1, cos_2, sin_0, sin_1, sin_2]
    #   3. Reshape for broadcast compatibility with attention heads
    raise NotImplementedError(
        "Must be extracted from codegen main_const_eval_22 (lines 658-1776)"
    )


def run_const_evals(weights, device):
    """Apply all const-eval transformations to weights dict.

    Args:
        weights: Dict mapping weight names to host TTNN tensors.
        device: TTNN device.

    Returns:
        Updated weights dict with transformed tensors.
    """
    # This function iterates over all weights and applies the appropriate
    # const-eval pattern based on the weight's role:
    #
    # - 2D weight matrices (*.weight with 2 dims): transpose_and_cast_to_f32
    # - 1D bias vectors (*.bias): cast_to_f32
    # - RoPE buffers: reshape_to_tile
    # - Attention masks: create_attn_mask
    # - Scalar constants: create_scalar_ones
    #
    # The exact mapping is determined by weight name patterns and the
    # arg_mapping.json configuration.
    raise NotImplementedError("Wire up after weight loading is implemented")
```

**Note:** The `precompute_rope_and_pos` and `run_const_evals` functions are stubs. They will be fully implemented in Task 7 (embedding modules) and Task 9 (integration), once we understand the exact data flow. The complex RoPE function (150+ ops) must be carefully extracted from the codegen.

- [ ] **Step 2: Commit**

```bash
git add z-transformer-only/consteval.py
git commit -m "refactor(z-transformer): add consteval.py with deduplicated patterns"
```

---

### Task 4: Create `model_ttnn.py` with weight loading

**Files:**
- Create: `z-transformer-only/model_ttnn.py`

This task creates the file skeleton with the weight loading function. The actual module classes are added in subsequent tasks.

- [ ] **Step 1: Write initial `model_ttnn.py` with weight loading**

The weight loading must convert each PyTorch tensor to the correct TTNN format. From analysis of `load_inputs_for__main` (lines 30008-35214), there are 5 loading configurations. Use `arg_mapping.json` to determine which config each weight needs.

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TTNN model for Z-Image Transformer with modular LightweightModule structure."""

import json
from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
)
COMPUTE_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
RMS_EPS = 9.9999997473787516e-06


def load_weights_from_pytorch(state_dict, device):
    """Convert PyTorch state_dict to TTNN tensors with correct layouts.

    Each weight is converted based on its role:
    - 2D matrices on device in TILE layout (BFLOAT16)
    - 1D biases kept on host in ROW_MAJOR (BFLOAT16)
    - Special cases: freqs (FLOAT32), position IDs (INT32)

    Args:
        state_dict: PyTorch model state_dict.
        device: TTNN device.

    Returns:
        Dict mapping weight names to TTNN tensors.
    """
    weights = {}

    for name, param in state_dict.items():
        tensor = ttnn.from_torch(param)

        if name == "t_embedder.freqs":
            # Float32 buffer, ROW_MAJOR, on device
            tensor = ttnn.to_dtype(tensor, ttnn.DataType.FLOAT32)
            tensor = ttnn.to_device(tensor, device, DRAM)
            weights[name] = tensor

        elif name.endswith("_pos_ids"):
            # Int32 position IDs, ROW_MAJOR, on host
            tensor = ttnn.to_dtype(tensor, ttnn.DataType.INT32)
            weights[name] = tensor

        elif name.endswith("_pad_mask") or name.endswith("_attn_mask"):
            # Masks: BFLOAT16, ROW_MAJOR, on host (const-eval transforms later)
            tensor = ttnn.to_dtype(tensor, ttnn.DataType.BFLOAT16)
            weights[name] = tensor

        elif param.dim() == 1 and any(
            name.endswith(s)
            for s in [".bias", ".weight"]
            if "norm" in name or "bias" in name
        ):
            # 1D norms and biases: check if they go on device or stay on host
            # Most norm weights go on device in TILE
            # Most biases (linear) stay on host in ROW_MAJOR
            tensor = ttnn.to_dtype(tensor, ttnn.DataType.BFLOAT16)
            if _weight_on_device(name):
                tensor = ttnn.to_layout(tensor, ttnn.Layout.TILE)
                tensor = ttnn.to_device(tensor, device, DRAM)
            weights[name] = tensor

        else:
            # Default: BFLOAT16, TILE layout, on device
            tensor = ttnn.to_dtype(tensor, ttnn.DataType.BFLOAT16)
            tensor = ttnn.to_layout(tensor, ttnn.Layout.TILE)
            tensor = ttnn.to_device(tensor, device, DRAM)
            weights[name] = tensor

    return weights


def _weight_on_device(name):
    """Determine if a 1D weight should be placed on device.

    From codegen analysis:
    - RMSNorm weights: on device (TILE)
    - Linear biases: on host (ROW_MAJOR) — transformed by const-eval
    - Pad tokens: on device (TILE)
    - adaLN biases: on host (ROW_MAJOR) — transformed by const-eval
    """
    if "norm" in name and name.endswith(".weight"):
        return True
    if "pad_token" in name:
        return True
    return False
```

- [ ] **Step 2: Commit**

```bash
git add z-transformer-only/model_ttnn.py
git commit -m "refactor(z-transformer): add model_ttnn.py skeleton with weight loading"
```

---

### Task 5: Implement TTNN FeedForward module

**Files:**
- Modify: `z-transformer-only/model_ttnn.py`

From codegen analysis (e.g., lines 28887-29037 for layers.29):
- `w1(x)` with SiLU activation fused
- `w3(x)` without activation
- `w1_out * w3_out` element-wise
- `w2(product)` output projection

```
matmul(x, w1, activation="silu") → matmul(x, w3) → multiply → matmul(result, w2)
```

- [ ] **Step 1: Add FeedForward class to `model_ttnn.py`**

```python
class FeedForward(LightweightModule):
    """SiLU-gated feed-forward network: w2(silu(w1(x)) * w3(x)).

    Weights:
        w1.weight: [ffn_hidden, dim] — gate projection (with SiLU)
        w2.weight: [dim, ffn_hidden] — output projection
        w3.weight: [ffn_hidden, dim] — up projection
    """

    def __init__(self, weights, prefix):
        self.w1 = weights[f"{prefix}.w1.weight"]
        self.w2 = weights[f"{prefix}.w2.weight"]
        self.w3 = weights[f"{prefix}.w3.weight"]

    def forward(self, x):
        # x: [seq_len, dim] in FLOAT32 on device
        w1_out = ttnn.matmul(
            x,
            self.w1,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation="silu",
            compute_kernel_config=COMPUTE_CONFIG,
        )
        w3_out = ttnn.matmul(
            x,
            self.w3,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        product = ttnn.multiply(w1_out, w3_out, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ttnn.deallocate(w1_out, False)
        ttnn.deallocate(w3_out, False)

        out = ttnn.matmul(
            product,
            self.w2,
            transpose_a=False,
            transpose_b=False,
            memory_config=DRAM,
            dtype=ttnn.DataType.FLOAT32,
            program_config=None,
            activation=None,
            compute_kernel_config=COMPUTE_CONFIG,
        )
        ttnn.deallocate(product, False)
        return out
```

- [ ] **Step 2: Commit**

```bash
git add z-transformer-only/model_ttnn.py
git commit -m "refactor(z-transformer): add FeedForward TTNN module"
```

---

### Task 6: Implement TTNN Attention module

**Files:**
- Modify: `z-transformer-only/model_ttnn.py`

From codegen analysis, each attention block:
1. Q projection: `matmul(x, to_q.weight)` → reshape to `[1, seq, n_heads, head_dim]`
2. K projection: `matmul(x, to_k.weight)` → reshape
3. V projection: `matmul(x, to_v.weight)` → reshape
4. QK norm: `rms_norm(q, norm_q.weight)`, `rms_norm(k, norm_k.weight)`
5. RoPE: slice into real/imag pairs → rotate → concatenate
6. Permute to `[1, n_heads, seq, head_dim]`
7. Typecast to BFLOAT16
8. SDPA (4 calls — the codegen splits this, but we should try a single call first and verify PCC)
9. Concatenate heads
10. Output projection: `matmul(result, to_out.weight)`

**Critical:** The codegen makes 4 SDPA calls per block. This may be a compiler optimization or a workaround. We start with a single SDPA call and verify PCC. If PCC drops, we replicate the 4-call splitting.

- [ ] **Step 1: Add Attention class**

```python
class Attention(LightweightModule):
    """Self-attention with QK-norm and RoPE.

    Weights:
        to_q.weight, to_k.weight, to_v.weight, to_out.weight: [dim, dim]
        norm_q.weight, norm_k.weight: [head_dim]
    """

    def __init__(self, weights, prefix, n_heads=30, head_dim=128):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = 1.0 / (head_dim ** 0.5)  # 0.088388...

        self.to_q = weights[f"{prefix}.to_q.weight"]
        self.to_k = weights[f"{prefix}.to_k.weight"]
        self.to_v = weights[f"{prefix}.to_v.weight"]
        self.to_out = weights[f"{prefix}.to_out.weight"]
        self.norm_q = weights[f"{prefix}.norm_q.weight"]
        self.norm_k = weights[f"{prefix}.norm_k.weight"]

    def forward(self, x, attn_mask, freqs_cis):
        """
        Args:
            x: [seq_len, dim] or [1, seq_len, dim] in FLOAT32
            attn_mask: [1, 1, seq_len, seq_len] attention mask
            freqs_cis: RoPE embeddings for this sequence
        Returns:
            [seq_len, dim] tensor
        """
        seq_len = x.shape[-2] if x.ndim == 3 else x.shape[0]
        dim = self.n_heads * self.head_dim

        # Q/K/V projections
        q = ttnn.matmul(x, self.to_q, transpose_a=False, transpose_b=False,
                        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
                        program_config=None, activation=None,
                        compute_kernel_config=COMPUTE_CONFIG)
        k = ttnn.matmul(x, self.to_k, transpose_a=False, transpose_b=False,
                        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
                        program_config=None, activation=None,
                        compute_kernel_config=COMPUTE_CONFIG)
        v = ttnn.matmul(x, self.to_v, transpose_a=False, transpose_b=False,
                        memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
                        program_config=None, activation=None,
                        compute_kernel_config=COMPUTE_CONFIG)

        # Reshape to [1, seq, n_heads, head_dim]
        q = ttnn.reshape(q, [1, seq_len, self.n_heads, self.head_dim], memory_config=DRAM)
        k = ttnn.reshape(k, [1, seq_len, self.n_heads, self.head_dim], memory_config=DRAM)
        v = ttnn.reshape(v, [1, seq_len, self.n_heads, self.head_dim], memory_config=DRAM)

        # QK norm
        q = ttnn.rms_norm(q, epsilon=RMS_EPS, weight=self.norm_q, bias=None,
                          residual_input_tensor=None, memory_config=DRAM,
                          program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM)
        k = ttnn.rms_norm(k, epsilon=RMS_EPS, weight=self.norm_k, bias=None,
                          residual_input_tensor=None, memory_config=DRAM,
                          program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM)

        # RoPE application
        q = self._apply_rope(q, freqs_cis)
        k = self._apply_rope(k, freqs_cis)

        # Permute to [1, n_heads, seq, head_dim] for SDPA
        q = ttnn.permute(q, [0, 2, 1, 3], memory_config=DRAM, pad_value=0.0)
        k = ttnn.permute(k, [0, 2, 1, 3], memory_config=DRAM, pad_value=0.0)
        v = ttnn.permute(v, [0, 2, 1, 3], memory_config=DRAM, pad_value=0.0)

        # Cast to BFLOAT16 for SDPA
        q = ttnn.typecast(q, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        k = ttnn.typecast(k, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        v = ttnn.typecast(v, ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # Scaled dot-product attention
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=self.scale,
        )

        # Concatenate heads: [1, n_heads, seq, head_dim] → [1, seq, dim]
        attn_out = ttnn.transformer.concatenate_heads(attn_out)

        # Reshape for output projection
        attn_out = ttnn.reshape(attn_out, [seq_len, dim], memory_config=DRAM)

        # Output projection
        out = ttnn.matmul(attn_out, self.to_out, transpose_a=False, transpose_b=False,
                          memory_config=DRAM, dtype=ttnn.DataType.FLOAT32,
                          program_config=None, activation=None,
                          compute_kernel_config=COMPUTE_CONFIG)
        ttnn.deallocate(attn_out, False)
        return out

    def _apply_rope(self, x, freqs_cis):
        """Apply real-valued rotary position embedding.

        x: [1, seq, n_heads, head_dim]
        freqs_cis: contains cos and sin components [seq, head_dim//2] each

        Rotary embedding splits head_dim into pairs (real, imag):
            out_real = x_real * cos - x_imag * sin
            out_imag = x_real * sin + x_imag * cos
        """
        # The exact RoPE application sequence must match the codegen.
        # From analysis (lines 26183-26425 for layer 29):
        #   1. Reshape x to [..., head_dim//2, 2] to split real/imag
        #   2. Slice out real (x[..., 0]) and imag (x[..., 1])
        #   3. Multiply by cos/sin and add/subtract
        #   4. Stack and reshape back
        #
        # This is extracted from the codegen and must match exactly.
        # Implementation details depend on how freqs_cis is structured,
        # which is determined by precompute_rope_and_pos in consteval.py.
        raise NotImplementedError(
            "RoPE application must be extracted from codegen lines 26183-26425"
        )
```

**Note:** The `_apply_rope` method is a stub. The exact implementation must be carefully extracted from the codegen's RoPE sequence (which involves reshape → slice → multiply → subtract/add → stack → reshape). This will be completed in Task 7 alongside the RoPE precomputation.

- [ ] **Step 2: Commit**

```bash
git add z-transformer-only/model_ttnn.py
git commit -m "refactor(z-transformer): add Attention TTNN module"
```

---

### Task 7: Implement RoPE and embedding modules

**Files:**
- Modify: `z-transformer-only/model_ttnn.py`
- Modify: `z-transformer-only/consteval.py`

This is the most complex task. It covers:
1. RoPE precomputation (`precompute_rope_and_pos` in consteval.py)
2. RoPE application (`Attention._apply_rope`)
3. TimestepEmbedder
4. Patch embedding (x_embedder)
5. Caption embedding (cap_embedder)

All sequences must be extracted from the codegen with exact op parameters.

- [ ] **Step 1: Implement `precompute_rope_and_pos` in `consteval.py`**

Extract the exact op sequence from `main_const_eval_22` (lines 658-1776 of the original main.py). This function:
- Takes 8 inputs: `[cap_pos_ids, sin_2, sin_1, sin_0, cos_2, cos_1, cos_0, image_pos_ids]`
- Performs index gathering on cos/sin tables using position IDs
- Concatenates across 3 axes
- Returns 6 outputs used as freqs_cis for image and caption tokens

The exact ops must be copied from the codegen. Read lines 658-1776 carefully and translate variable names.

- [ ] **Step 2: Implement `Attention._apply_rope`**

Extract from codegen lines 26183-26425. The pattern is:
```
reshape [1, seq, heads, head_dim] → [1, seq, heads, head_dim//2, 2]
slice [..., 0] → x_real
slice [..., 1] → x_imag
x_real * cos - x_imag * sin → out_real
x_real * sin + x_imag * cos → out_imag
stack and reshape back to [1, seq, heads, head_dim]
```

- [ ] **Step 3: Add TimestepEmbedder to `model_ttnn.py`**

From codegen lines 4426-4534:
```python
class TimestepEmbedder(LightweightModule):
    """Timestep → embedding via frequency encoding + 2-layer MLP.

    freqs * t → [cos, sin] → Linear+SiLU → Linear → [1, adaln_dim]
    """

    def __init__(self, weights):
        self.freqs = weights["t_embedder.freqs"]
        self.mlp_0_weight = weights["t_embedder.mlp.0.weight"]
        self.mlp_0_bias = weights["t_embedder.mlp.0.bias"]
        self.mlp_2_weight = weights["t_embedder.mlp.2.weight"]
        self.mlp_2_bias = weights["t_embedder.mlp.2.bias"]

    def forward(self, t):
        """t: scalar timestep tensor [1, 1] in FLOAT32."""
        t = ttnn.to_layout(t, ttnn.Layout.TILE, None, memory_config=None)
        t = ttnn.reshape(t, [1, 1], memory_config=DRAM)

        # Frequency encoding
        freqs = ttnn.multiply(t, self.freqs, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
        cos_t = ttnn.cos(freqs, memory_config=DRAM)
        sin_t = ttnn.sin(freqs, memory_config=DRAM)
        t_emb = ttnn.concat([cos_t, sin_t], 1, memory_config=DRAM)

        # MLP: Linear(256, 1024) + SiLU
        x = ttnn.linear(
            t_emb, self.mlp_0_weight, bias=self.mlp_0_bias,
            transpose_a=False, transpose_b=False, memory_config=DRAM,
            dtype=ttnn.DataType.FLOAT32, program_config=None,
            activation="silu", compute_kernel_config=COMPUTE_CONFIG,
        )
        # MLP: Linear(1024, 256)
        x = ttnn.linear(
            x, self.mlp_2_weight, bias=self.mlp_2_bias,
            transpose_a=False, transpose_b=False, memory_config=DRAM,
            dtype=ttnn.DataType.FLOAT32, program_config=None,
            activation=None, compute_kernel_config=COMPUTE_CONFIG,
        )
        return x  # [1, 256] FLOAT32
```

- [ ] **Step 4: Add embedding helper methods to top-level model**

The patch embedding and caption embedding pipelines (codegen lines 4313-4600) will be methods on the top-level `ZImageTransformerTTNN` class. Document the exact sequences here for Task 9.

- [ ] **Step 5: Commit**

```bash
git add z-transformer-only/model_ttnn.py z-transformer-only/consteval.py
git commit -m "refactor(z-transformer): add RoPE, TimestepEmbedder, and embedding helpers"
```

---

### Task 8: Implement TransformerBlock and FinalLayer

**Files:**
- Modify: `z-transformer-only/model_ttnn.py`

- [ ] **Step 1: Add TransformerBlock class**

From codegen analysis, each block follows this exact sequence:
```
adaLN: linear(t_emb, adaln_weight, adaln_bias) → typecast → reshape [1,1,15360]
       → slice [0:3840] for scale_msa, [3840:7680] for gate_msa (tanh),
         [7680:11520] for scale_mlp, [11520:15360] for gate_mlp (tanh)

attention path:
  rms_norm(x, attention_norm1) → multiply by (1 + scale_msa) → attention →
  rms_norm(attn_out, attention_norm2) → multiply by gate_msa → add to residual

ffn path:
  rms_norm(x, ffn_norm1) → multiply by (1 + scale_mlp) → feed_forward →
  rms_norm(ff_out, ffn_norm2) → multiply by gate_mlp → add to residual
```

```python
class TransformerBlock(LightweightModule):
    """Transformer block with optional adaLN modulation.

    Two variants:
    - With modulation (noise_refiner, main layers): uses adaLN for scale/gate
    - Without modulation (context_refiner): direct norm → attention → residual
    """

    def __init__(self, weights, prefix, n_heads=30, head_dim=128, modulation=True):
        self.modulation = modulation
        self.attention = Attention(weights, f"{prefix}.attention", n_heads, head_dim)
        self.feed_forward = FeedForward(weights, f"{prefix}.feed_forward")

        self.attention_norm1 = weights[f"{prefix}.attention_norm1.weight"]
        self.attention_norm2 = weights[f"{prefix}.attention_norm2.weight"]
        self.ffn_norm1 = weights[f"{prefix}.ffn_norm1.weight"]
        self.ffn_norm2 = weights[f"{prefix}.ffn_norm2.weight"]

        if modulation:
            self.adaln_weight = weights[f"{prefix}.adaLN_modulation.0.weight"]
            self.adaln_bias = weights[f"{prefix}.adaLN_modulation.0.bias"]

    def forward(self, x, attn_mask, freqs_cis, adaln_input=None):
        """
        Args:
            x: [1, seq_len, dim] hidden states
            attn_mask: attention mask
            freqs_cis: RoPE embeddings
            adaln_input: [1, adaln_dim] timestep embedding (only if modulation=True)
        """
        seq_len = x.shape[1]
        dim = x.shape[2]

        if self.modulation:
            # adaLN modulation
            mod = ttnn.linear(
                adaln_input, self.adaln_weight, bias=self.adaln_bias,
                transpose_a=False, transpose_b=False, memory_config=DRAM,
                dtype=ttnn.DataType.FLOAT32, program_config=None,
                activation=None, compute_kernel_config=COMPUTE_CONFIG,
            )
            mod = ttnn.typecast(mod, ttnn.DataType.BFLOAT16, memory_config=DRAM)
            mod = ttnn.reshape(mod, [1, 1, 4 * dim], memory_config=DRAM)

            # Split into 4 chunks: scale_msa, gate_msa, scale_mlp, gate_mlp
            scale_msa = ttnn.slice(mod, [0, 0, 0], [1, 1, dim], [1, 1, 1], memory_config=DRAM)
            gate_msa = ttnn.slice(mod, [0, 0, dim], [1, 1, 2 * dim], [1, 1, 1], memory_config=DRAM)
            scale_mlp = ttnn.slice(mod, [0, 0, 2 * dim], [1, 1, 3 * dim], [1, 1, 1], memory_config=DRAM)
            gate_mlp = ttnn.slice(mod, [0, 0, 3 * dim], [1, 1, 4 * dim], [1, 1, 1], memory_config=DRAM)
            gate_msa = ttnn.tanh(gate_msa, fast_and_approximate_mode=False, memory_config=DRAM)
            gate_mlp = ttnn.tanh(gate_mlp, fast_and_approximate_mode=False, memory_config=DRAM)

            # Attention path
            x_norm = ttnn.rms_norm(
                x, epsilon=RMS_EPS, weight=self.attention_norm1, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            # scale_msa = 1 + scale_msa
            ones = ttnn.full(shape=scale_msa.shape, fill_value=1.0,
                             dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE,
                             device=x.device(), memory_config=DRAM)
            scale_msa = ttnn.add(scale_msa, ones, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            x_scaled = ttnn.multiply(x_norm, scale_msa, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

            # Reshape for attention: [1, seq, dim] → [seq, dim]
            x_flat = ttnn.reshape(x_scaled, [seq_len, dim], memory_config=DRAM)
            attn_out = self.attention(x_flat, attn_mask, freqs_cis)
            attn_out = ttnn.reshape(attn_out, [1, seq_len, dim], memory_config=DRAM)

            attn_out_norm = ttnn.rms_norm(
                attn_out, epsilon=RMS_EPS, weight=self.attention_norm2, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            x = ttnn.add(x, ttnn.multiply(gate_msa, attn_out_norm,
                         dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM),
                         dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

            # FFN path
            x_norm2 = ttnn.rms_norm(
                x, epsilon=RMS_EPS, weight=self.ffn_norm1, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            scale_mlp = ttnn.add(scale_mlp, ones, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
            x_scaled2 = ttnn.multiply(x_norm2, scale_mlp, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

            x_flat2 = ttnn.reshape(x_scaled2, [seq_len, dim], memory_config=DRAM)
            ff_out = self.feed_forward(x_flat2)
            ff_out = ttnn.reshape(ff_out, [1, seq_len, dim], memory_config=DRAM)

            ff_out_norm = ttnn.rms_norm(
                ff_out, epsilon=RMS_EPS, weight=self.ffn_norm2, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            x = ttnn.add(x, ttnn.multiply(gate_mlp, ff_out_norm,
                         dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM),
                         dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)
        else:
            # No modulation (context_refiner blocks)
            x_norm = ttnn.rms_norm(
                x, epsilon=RMS_EPS, weight=self.attention_norm1, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            x_flat = ttnn.reshape(x_norm, [seq_len, dim], memory_config=DRAM)
            attn_out = self.attention(x_flat, attn_mask, freqs_cis)
            attn_out = ttnn.reshape(attn_out, [1, seq_len, dim], memory_config=DRAM)
            attn_out_norm = ttnn.rms_norm(
                attn_out, epsilon=RMS_EPS, weight=self.attention_norm2, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            x = ttnn.add(x, attn_out_norm, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

            x_norm2 = ttnn.rms_norm(
                x, epsilon=RMS_EPS, weight=self.ffn_norm1, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            x_flat2 = ttnn.reshape(x_norm2, [seq_len, dim], memory_config=DRAM)
            ff_out = self.feed_forward(x_flat2)
            ff_out = ttnn.reshape(ff_out, [1, seq_len, dim], memory_config=DRAM)
            ff_out_norm = ttnn.rms_norm(
                ff_out, epsilon=RMS_EPS, weight=self.ffn_norm2, bias=None,
                residual_input_tensor=None, memory_config=DRAM,
                program_config=None, compute_kernel_config=COMPUTE_CONFIG_NORM,
            )
            x = ttnn.add(x, ff_out_norm, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

        return x
```

- [ ] **Step 2: Add FinalLayer class**

From codegen lines 29894-30005:
```python
class FinalLayer(LightweightModule):
    """Output projection with adaLN modulation.

    adaLN_modulation(silu(t_emb)) → scale
    layer_norm(x) * (1 + scale) → linear → output
    Then un-patchify: extract image tokens, reshape to [C, F, H, W].
    """

    def __init__(self, weights):
        self.linear_weight = weights["final_layer.linear.weight"]
        self.linear_bias = weights["final_layer.linear.bias"]
        self.adaln_weight = weights["final_layer.adaLN_modulation.1.weight"]
        self.adaln_bias = weights["final_layer.adaLN_modulation.1.bias"]

    def forward(self, x, adaln_input, image_ori_len=3600):
        """
        Args:
            x: [1, total_seq, dim] — full sequence (image + caption)
            adaln_input: [1, adaln_dim] timestep embedding
            image_ori_len: number of original (non-padded) image tokens
        Returns:
            [C, F, H, W] = [16, 1, 160, 90] output tensor
        """
        # adaLN modulation
        t = ttnn.silu(adaln_input, memory_config=DRAM)
        scale = ttnn.linear(
            t, self.adaln_weight, bias=self.adaln_bias,
            transpose_a=False, transpose_b=False, memory_config=DRAM,
            dtype=ttnn.DataType.FLOAT32, program_config=None,
            activation=None, compute_kernel_config=COMPUTE_CONFIG,
        )
        scale = ttnn.typecast(scale, ttnn.DataType.BFLOAT16, memory_config=DRAM)
        ones = ttnn.full(shape=scale.shape, fill_value=1.0,
                         dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE,
                         device=x.device(), memory_config=DRAM)
        scale = ttnn.add(scale, ones, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # Layer norm (not RMS norm) — codegen uses typecast to float32 + manual norm
        # TODO: verify if this is layer_norm or a manual implementation
        x_normed = ttnn.multiply(x, scale, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # Linear projection
        out = ttnn.matmul(
            x_normed, self.linear_weight,
            transpose_a=False, transpose_b=True, memory_config=DRAM,
            dtype=ttnn.DataType.BFLOAT16, program_config=None,
            activation=None, compute_kernel_config=COMPUTE_CONFIG,
        )
        out = ttnn.add(out, self.linear_bias, dtype=ttnn.DataType.BFLOAT16, memory_config=DRAM)

        # Extract image tokens only (first image_ori_len)
        out = ttnn.slice(out, [0, 0, 0], [1, image_ori_len, 64], [1, 1, 1], memory_config=DRAM)

        # Un-patchify: [1, 3600, 64] → [1, 80, 45, 1, 2, 2, 16] → permute → [16, 1, 160, 90]
        out = ttnn.reshape(out, [1, 80, 45, 1, 2, 2, 16], memory_config=DRAM)
        out = ttnn.permute(out, [6, 0, 3, 1, 4, 2, 5], memory_config=DRAM, pad_value=0.0)
        out = ttnn.reshape(out, [16, 1, 160, 90], memory_config=DRAM)

        return out
```

- [ ] **Step 3: Commit**

```bash
git add z-transformer-only/model_ttnn.py
git commit -m "refactor(z-transformer): add TransformerBlock and FinalLayer TTNN modules"
```

---

### Task 9: Implement top-level ZImageTransformerTTNN and wire everything together

**Files:**
- Modify: `z-transformer-only/model_ttnn.py`
- Modify: `z-transformer-only/consteval.py`

- [ ] **Step 1: Add ZImageTransformerTTNN class**

```python
class ZImageTransformerTTNN(LightweightModule):
    """Z-Image Transformer in TTNN.

    Architecture:
        - TimestepEmbedder: timestep → adaln_input [1, 256]
        - Patch embedding: latent [16,1,160,90] → image tokens [1, 3616, 3840]
        - Caption embedding: cap_feat [18, 2560] → cap tokens [1, 32, 3840]
        - RoPE precomputation
        - noise_refiner: 2 TransformerBlocks (image only, with modulation)
        - context_refiner: 2 TransformerBlocks (caption only, no modulation)
        - Concatenate image + caption → unified sequence [1, 3648, 3840]
        - layers: 30 TransformerBlocks (unified, with modulation)
        - FinalLayer → output [16, 1, 160, 90]
    """

    def __init__(self, device, torch_state_dict):
        self.device = device

        # Load weights from PyTorch
        weights = load_weights_from_pytorch(torch_state_dict, device)

        # Apply const-eval transformations
        weights = run_const_evals(weights, device)

        # Build modules
        self.t_embedder = TimestepEmbedder(weights)
        self.final_layer = FinalLayer(weights)

        self.noise_refiner = [
            TransformerBlock(weights, f"noise_refiner.{i}", modulation=True)
            for i in range(2)
        ]
        self.context_refiner = [
            TransformerBlock(weights, f"context_refiner.{i}", modulation=False)
            for i in range(2)
        ]
        self.layers = [
            TransformerBlock(weights, f"layers.{i}", modulation=True)
            for i in range(30)
        ]

        # Store embedding weights
        self.x_embedder_weight = weights["x_embedder.weight"]
        self.x_embedder_bias = weights["x_embedder.bias"]
        self.cap_embedder_norm_weight = weights["cap_embedder.0.weight"]
        self.cap_embedder_linear_weight = weights["cap_embedder.1.weight"]
        self.cap_embedder_linear_bias = weights["cap_embedder.1.bias"]
        self.x_pad_token = weights["x_pad_token"]
        self.cap_pad_token = weights["cap_pad_token"]

        # Store buffer tensors
        self.image_pad_mask = weights["image_pad_mask"]
        self.cap_pad_mask = weights["cap_pad_mask"]
        self.x_attn_mask = weights["x_attn_mask"]
        self.cap_attn_mask = weights["cap_attn_mask"]
        self.unified_attn_mask = weights["unified_attn_mask"]
        self.image_pos_ids = weights["image_pos_ids"]
        self.cap_pos_ids = weights["cap_pos_ids"]
        self.rope_cos_0 = weights["rope_embedder.cos_0"]
        self.rope_cos_1 = weights["rope_embedder.cos_1"]
        self.rope_cos_2 = weights["rope_embedder.cos_2"]
        self.rope_sin_0 = weights["rope_embedder.sin_0"]
        self.rope_sin_1 = weights["rope_embedder.sin_1"]
        self.rope_sin_2 = weights["rope_embedder.sin_2"]

    def forward(self, latent_input, timestep, cap_feat):
        """
        Args:
            latent_input: [16, 1, 160, 90] BFLOAT16 host tensor
            timestep: [1] FLOAT32 host tensor
            cap_feat: [18, 2560] BFLOAT16 host tensor

        Returns:
            [16, 1, 160, 90] output tensor on device
        """
        # 1. Timestep embedding
        adaln_input = self.t_embedder(timestep)  # [1, 256] FLOAT32

        # 2. Patch embedding
        image_tokens = self._embed_patches(latent_input)  # [1, 3616, 3840]

        # 3. Caption embedding
        cap_tokens = self._embed_caption(cap_feat)  # [1, 32, 3840]

        # 4. Precompute RoPE
        image_freqs, cap_freqs, unified_freqs = self._precompute_rope()

        # 5. Noise refiner (image tokens only)
        for block in self.noise_refiner:
            image_tokens = block(image_tokens, self.x_attn_mask, image_freqs, adaln_input)

        # 6. Context refiner (caption tokens only)
        for block in self.context_refiner:
            cap_tokens = block(cap_tokens, self.cap_attn_mask, cap_freqs)

        # 7. Concatenate image + caption
        x = ttnn.concat([image_tokens, cap_tokens], 1, memory_config=DRAM)

        # 8. Main transformer layers
        for block in self.layers:
            x = block(x, self.unified_attn_mask, unified_freqs, adaln_input)

        # 9. Final layer
        output = self.final_layer(x, adaln_input)

        return output

    def _embed_patches(self, latent_input):
        """Convert latent [16,1,160,90] → image tokens [1, 3616, 3840].

        Exact sequence from codegen lines 4313-4425.
        """
        # Must be extracted from codegen:
        # to_layout → reshape [16,1,1,80,2,45,2] → permute [1,3,5,2,4,6,0]
        # → reshape [3600,64] → pad with x_pad_token → reshape [1,3616,3840]
        raise NotImplementedError("Extract from codegen lines 4313-4425")

    def _embed_caption(self, cap_feat):
        """Convert caption [18,2560] → caption tokens [1, 32, 3840].

        Exact sequence from codegen lines 4598-4650.
        """
        # Must be extracted from codegen:
        # rms_norm(cap_feat, cap_embedder_norm_weight) → linear → pad with cap_pad_token
        raise NotImplementedError("Extract from codegen lines 4598-4650")

    def _precompute_rope(self):
        """Precompute RoPE for image, caption, and unified sequences."""
        # Calls precompute_rope_and_pos from consteval.py
        raise NotImplementedError("Wire up consteval.precompute_rope_and_pos")
```

- [ ] **Step 2: Implement `run_const_evals` in `consteval.py`**

Wire up the const-eval function that transforms weights based on their role. Each weight name maps to a specific const-eval pattern based on the analysis in `arg_mapping.json`.

- [ ] **Step 3: Implement the `_embed_patches`, `_embed_caption`, and `_precompute_rope` methods**

Extract exact op sequences from the codegen. These are the remaining stubs that need real implementations.

- [ ] **Step 4: Commit**

```bash
git add z-transformer-only/model_ttnn.py z-transformer-only/consteval.py
git commit -m "refactor(z-transformer): add ZImageTransformerTTNN top-level module"
```

---

### Task 10: Create new `main.py` and run PCC verification

**Files:**
- Create: `z-transformer-only/main.py` (replace the 35K-line codegen)

- [ ] **Step 1: Back up the original codegen main.py**

```bash
mv z-transformer-only/main.py z-transformer-only/main_codegen.py
```

- [ ] **Step 2: Write new `main.py`**

```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import time

import torch
import ttnn
from model_pt import ZImageTransformerPT, get_input
from model_ttnn import ZImageTransformerTTNN
from utils import calculate_pcc


def main():
    """
    Z-Image Transformer (Tongyi-MAI/Z-Image).

    Runs the model on both PyTorch (CPU) and TTNN (TT device),
    compares PCC to ensure numerical accuracy is preserved.
    """

    # Load PyTorch model and get reference output
    model_pt = ZImageTransformerPT()
    latent_input, timestep, cap_feat = get_input(model_pt)

    print("Running PyTorch reference...")
    with torch.inference_mode():
        output_pt = model_pt(latent_input, timestep, cap_feat)
    print(f"PyTorch output shape: {output_pt.shape}")

    # Convert inputs to TTNN host tensors
    latent_ttnn = ttnn.from_torch(latent_input)
    latent_ttnn = ttnn.to_dtype(latent_ttnn, ttnn.DataType.BFLOAT16)
    timestep_ttnn = ttnn.from_torch(timestep)
    timestep_ttnn = ttnn.to_dtype(timestep_ttnn, ttnn.DataType.FLOAT32)
    cap_feat_ttnn = ttnn.from_torch(cap_feat)
    cap_feat_ttnn = ttnn.to_dtype(cap_feat_ttnn, ttnn.DataType.BFLOAT16)

    # Open device
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )

    # Load TTNN model from PyTorch weights
    model_ttnn = ZImageTransformerTTNN(mesh_device, model_pt.state_dict_for_ttnn())

    # Run TTNN model
    for i in range(3):
        start_time = time.time()

        out_device = model_ttnn(latent_ttnn, timestep_ttnn, cap_feat_ttnn)
        out_host = ttnn.from_device(out_device, blocking=True)
        ttnn.synchronize_device(mesh_device)

        end_time = time.time()

        duration = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time)
        pcc = calculate_pcc(output_pt, ttnn.to_torch(out_host))

        print(f"Iteration {i}")
        print(f"\tDuration: {duration:.1f}ms")
        print(f"\tFPS: {fps:.2f}")
        print(f"\tPCC: {pcc:.6f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run PCC verification**

```bash
cd z-transformer-only && ./run
```

Expected output:
```
PyTorch output shape: torch.Size([16, 1, 160, 90])
Iteration 0
    Duration: ~XXXXms
    PCC: 1.000004 (must be > 0.99)
```

If PCC < 0.99, investigate:
1. Check weight loading — compare TTNN weights against codegen tensorbin values
2. Check const-eval transformations
3. Check individual module outputs against codegen intermediate tensors
4. Check attention SDPA — may need to replicate 4-call splitting

- [ ] **Step 4: Commit**

```bash
git add z-transformer-only/main.py
git commit -m "refactor(z-transformer): add clean main.py with PCC verification"
```

---

### Task 11: Fill in all remaining stubs

**Files:**
- Modify: `z-transformer-only/model_ttnn.py`
- Modify: `z-transformer-only/consteval.py`

This task fills in every `raise NotImplementedError` stub by extracting exact op sequences from the codegen (`main_codegen.py`). Each stub has a comment indicating which codegen lines to extract from.

- [ ] **Step 1: Implement `Attention._apply_rope`**

Read `main_codegen.py` lines 26183-26425. Extract the exact RoPE sequence:
- reshape to split head_dim into pairs
- slice real/imag
- multiply by cos/sin
- subtract/add
- reshape back

- [ ] **Step 2: Implement `precompute_rope_and_pos` in consteval.py**

Read `main_codegen.py` lines 658-1776 (main_const_eval_22). This is the largest extraction — 150+ ops that compute position embeddings from cos/sin lookup tables. Group ops by function (gather, concatenate, reshape) and simplify variable names.

- [ ] **Step 3: Implement `_embed_patches`**

Read `main_codegen.py` lines 4313-4425.

- [ ] **Step 4: Implement `_embed_caption`**

Read `main_codegen.py` lines 4598-4650.

- [ ] **Step 5: Implement `_precompute_rope`**

Wire up the precompute_rope_and_pos call with the correct buffer tensors.

- [ ] **Step 6: Implement `run_const_evals`**

Build the weight transformation pipeline. For each weight, determine which const-eval pattern applies based on its name and role.

- [ ] **Step 7: Run full model and verify PCC**

```bash
cd z-transformer-only && ./run
```

Expected: PCC > 0.99

- [ ] **Step 8: Commit**

```bash
git add z-transformer-only/model_ttnn.py z-transformer-only/consteval.py
git commit -m "refactor(z-transformer): implement all remaining stubs"
```

---

### Task 12: Debug and fix PCC mismatches

**Files:**
- Modify: `z-transformer-only/model_ttnn.py`
- Modify: `z-transformer-only/consteval.py`

If PCC is not matching after Task 11, this task provides a systematic debugging approach.

- [ ] **Step 1: Add intermediate tensor comparison**

Modify `main.py` to also run the old codegen and compare intermediate tensors:

```python
# In main.py, temporarily:
import main_codegen
codegen_inputs = main_codegen.load_inputs_for__main()
codegen_outputs = main_codegen._main(codegen_inputs)
# Compare codegen_outputs vs model_ttnn outputs
```

- [ ] **Step 2: Compare weight loading**

For each weight, compare the TTNN tensor from `load_weights_from_pytorch` against the corresponding tensorbin loaded by the codegen. They should be bit-identical.

```python
import ttnn
# Load from codegen
codegen_tensor = ttnn.load_tensor("tensors/arg10.tensorbin")
# Load from our pipeline
our_tensor = weights["layers.29.ffn_norm2.weight"]
# Compare
pcc = calculate_pcc(ttnn.to_torch(codegen_tensor), ttnn.to_torch(our_tensor))
print(f"Weight PCC: {pcc}")
```

- [ ] **Step 3: Compare after const-eval**

Verify that const-eval transformed tensors match the codegen's cached const-eval results.

- [ ] **Step 4: Compare after each module**

Insert tensor dumps after TimestepEmbedder, after patch embedding, after first transformer block, etc. Compare against corresponding points in the codegen execution.

- [ ] **Step 5: Fix any mismatches found**

Common issues:
- Weight transpose direction (transpose_a vs transpose_b)
- Missing typecast (BFLOAT16 vs FLOAT32)
- Shape mismatch in reshape
- Attention mask format
- RoPE axis ordering

- [ ] **Step 6: Verify PCC**

```bash
cd z-transformer-only && ./run
```

Expected: PCC > 0.99 (ideally 1.0000042915 matching baseline)

- [ ] **Step 7: Commit**

```bash
git add z-transformer-only/model_ttnn.py z-transformer-only/consteval.py
git commit -m "fix(z-transformer): fix PCC mismatches in refactored model"
```

---

### Task 13: Cleanup

**Files:**
- Delete: `z-transformer-only/main_codegen.py` (backup of original 35K-line file)
- Delete: `z-transformer-only/tensors/` directory (538 .tensorbin files)
- Delete: `z-transformer-only/ttnn.mlir`
- Delete: `z-transformer-only/irs/`
- Delete: `z-transformer-only/generated/`
- Delete: `z-transformer-only/__pycache__/`
- Delete: `z-transformer-only/__init__.py` (if no longer needed)
- Keep: `z-transformer-only/parse_mlir_args.py` and `z-transformer-only/arg_mapping.json` (useful reference)

- [ ] **Step 1: Run final PCC verification before cleanup**

```bash
cd z-transformer-only && ./run
```

Confirm PCC > 0.99.

- [ ] **Step 2: Remove codegen artifacts**

```bash
rm z-transformer-only/main_codegen.py
rm -rf z-transformer-only/tensors/
rm z-transformer-only/ttnn.mlir
rm -rf z-transformer-only/irs/
rm -rf z-transformer-only/generated/
rm -rf z-transformer-only/__pycache__/
rm z-transformer-only/__init__.py
```

- [ ] **Step 3: Verify the clean directory runs**

```bash
cd z-transformer-only && ./run
```

Expected: Same PCC as before cleanup.

- [ ] **Step 4: Final commit**

```bash
git add -A z-transformer-only/
git commit -m "refactor(z-transformer): remove codegen artifacts, refactoring complete"
```

---

## Execution Notes

### Iteration Speed

The full model takes ~158s to run. During development:
- Use the old codegen (`main_codegen.py`) as a reference for intermediate tensor values
- For quick checks, compare individual weights and const-eval outputs before running the full model
- Only run the full model for milestone PCC checks (after Task 10, 11, 12)

### Known Challenges

1. **RoPE extraction (Task 7/11)**: `main_const_eval_22` is 150+ ops. Requires careful extraction.
2. **Attention SDPA splitting**: The codegen uses 4 SDPA calls per block. Try single call first — if PCC drops, investigate whether the splitting was a correctness requirement or just compiler optimization.
3. **Weight format**: The codegen's const-evals transpose and cast weights. The `load_weights_from_pytorch` + `run_const_evals` pipeline must produce identical tensors.
4. **FinalLayer norm**: The codegen may use a manual layer_norm implementation rather than `ttnn.layer_norm`. Must match exactly.
