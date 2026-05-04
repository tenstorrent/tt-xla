# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for CogVideoX-5b component tests.

Exposes:
  - load_text_encoder / load_transformer / load_vae: HuggingFace model loaders
  - cogvideox_mesh: 2D ("batch", "model") SPMD mesh — adapts to device count
  - shard_text_encoder_specs / shard_transformer_specs / shard_vae_decoder_specs:
    per-component shard spec functions returning dict[Tensor, partition_spec]
    for run_graph_test.

Defaults follow the modified inference at
tests/torch/models/test_cog5x_num1.py (num_frames=9, num_inference_steps=1).
"""

import torch
import torch.nn as nn
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh

# ---------------------------------------------------------------------------
# Model + dtype
# ---------------------------------------------------------------------------

REPO_ID = "THUDM/CogVideoX-5b"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Default inference shape (matches test_cog5x_num1.py: num_frames=9)
# ---------------------------------------------------------------------------

HEIGHT = 480
WIDTH = 720
NUM_FRAMES = 9
MAX_SEQ_LEN = 226  # T5 text token length for CogVideoX

VAE_SCALE_SPATIAL = 8
VAE_SCALE_TEMPORAL = 4

LATENT_H = HEIGHT // VAE_SCALE_SPATIAL  # 60
LATENT_W = WIDTH // VAE_SCALE_SPATIAL  # 90
NUM_LATENT_FRAMES = (NUM_FRAMES - 1) // VAE_SCALE_TEMPORAL + 1  # 3
NUM_CHANNELS_LATENTS = 16
TEXT_EMBED_DIM = 4096
INNER_DIM = 1920  # num_attention_heads * attention_head_dim = 30 * 64


# ---------------------------------------------------------------------------
# Component loaders (real weights from HuggingFace)
# ---------------------------------------------------------------------------


def load_text_encoder():
    """Load the T5 text encoder for CogVideoX-5b."""
    from transformers import T5EncoderModel

    return T5EncoderModel.from_pretrained(
        REPO_ID,
        subfolder="text_encoder",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).eval()


def load_transformer(max_blocks: int = 0):
    """Load CogVideoXTransformer3DModel.

    If `max_blocks > 0`, truncate to that many transformer blocks (useful for
    compile-time debugging; the full CogVideoX-5b has 30 layers).
    """
    from diffusers import CogVideoXTransformer3DModel

    transformer = CogVideoXTransformer3DModel.from_pretrained(
        REPO_ID,
        subfolder="transformer",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    if max_blocks > 0 and len(transformer.transformer_blocks) > max_blocks:
        transformer.transformer_blocks = nn.ModuleList(
            list(transformer.transformer_blocks[:max_blocks])
        )
    return transformer.eval()


def load_vae():
    """Load AutoencoderKLCogVideoX (3D causal VAE)."""
    from diffusers import AutoencoderKLCogVideoX

    return AutoencoderKLCogVideoX.from_pretrained(
        REPO_ID,
        subfolder="vae",
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    ).eval()


# ---------------------------------------------------------------------------
# SPMD mesh
# ---------------------------------------------------------------------------

# (batch, model) shapes by device count
_MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}


def cogvideox_mesh() -> Mesh:
    """2D ("batch", "model") SPMD mesh sized to current device count."""
    import torch_xla.runtime as xr

    n = xr.global_runtime_device_count()
    if n not in _MESH_SHAPES:
        raise ValueError(
            f"Unsupported device count: {n}. "
            f"Expected one of {sorted(_MESH_SHAPES)}."
        )
    return get_mesh(_MESH_SHAPES[n], ("batch", "model"))


# ---------------------------------------------------------------------------
# T5 text encoder sharding
# ---------------------------------------------------------------------------


def shard_text_encoder_specs(encoder) -> dict:
    """Build shard specs for T5EncoderModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, wi/wi_0, wi_1):  ("model", "batch")
    Row-parallel   (o, wo):                    ("batch", "model")

    CogVideoX-5b's T5 uses the gated FFN (DenseReluDense with wi_0/wi_1/wo);
    we shard whichever projections are present.
    """
    specs = {encoder.shared.weight: (None, "batch")}

    for block in encoder.encoder.block:
        sa = block.layer[0].SelfAttention
        specs[sa.q.weight] = ("model", "batch")
        specs[sa.k.weight] = ("model", "batch")
        specs[sa.v.weight] = ("model", "batch")
        specs[sa.o.weight] = ("batch", "model")
        specs[block.layer[0].layer_norm.weight] = ("batch",)

        ffn = block.layer[1].DenseReluDense
        if hasattr(ffn, "wi_0"):
            specs[ffn.wi_0.weight] = ("model", "batch")
            specs[ffn.wi_1.weight] = ("model", "batch")
        else:
            specs[ffn.wi.weight] = ("model", "batch")
        specs[ffn.wo.weight] = ("batch", "model")
        specs[block.layer[1].layer_norm.weight] = ("batch",)

    specs[encoder.encoder.final_layer_norm.weight] = ("batch",)
    return specs


# ---------------------------------------------------------------------------
# CogVideoXTransformer3DModel sharding
# ---------------------------------------------------------------------------


def shard_transformer_specs(transformer) -> dict:
    """Build shard specs for CogVideoXTransformer3DModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (Q, K, V, FFN up):  ("model", "batch")
    Row-parallel   (O, FFN down):        ("batch", "model")
    """
    specs = {
        # Patch embedding: Conv2d(16, 1920, k=2x2) + Linear(4096, 1920) for text
        transformer.patch_embed.proj.weight: ("batch", None, None, None),
        transformer.patch_embed.proj.bias: ("batch",),
        transformer.patch_embed.text_proj.weight: ("batch", "model"),
        transformer.patch_embed.text_proj.bias: ("batch",),
    }

    # Time embedding: Linear(1920, 512) -> Linear(512, 512)
    specs[transformer.time_embedding.linear_1.weight] = ("model", "batch")
    specs[transformer.time_embedding.linear_1.bias] = ("model",)
    specs[transformer.time_embedding.linear_2.weight] = ("batch", "model")
    specs[transformer.time_embedding.linear_2.bias] = ("batch",)

    # Optional ofs embedding (only present in CogVideoX-1.5 I2V variants)
    if getattr(transformer, "ofs_embedding", None) is not None:
        specs[transformer.ofs_embedding.linear_1.weight] = ("model", "batch")
        specs[transformer.ofs_embedding.linear_1.bias] = ("model",)
        specs[transformer.ofs_embedding.linear_2.weight] = ("batch", "model")
        specs[transformer.ofs_embedding.linear_2.bias] = ("batch",)

    # Per-block sharding
    for block in transformer.transformer_blocks:
        # AdaLN-Zero conditioners: Linear(512, 6*1920) into the conditioning path
        for ada in (block.norm1, block.norm2):
            specs[ada.linear.weight] = ("model", "batch")
            specs[ada.linear.bias] = ("model",)
            if getattr(ada.norm, "weight", None) is not None:
                specs[ada.norm.weight] = ("batch",)
            if getattr(ada.norm, "bias", None) is not None:
                specs[ada.norm.bias] = ("batch",)

        # Self-attention (the only attention in CogVideoXBlock)
        attn = block.attn1
        specs[attn.to_q.weight] = ("model", "batch")
        specs[attn.to_k.weight] = ("model", "batch")
        specs[attn.to_v.weight] = ("model", "batch")
        if getattr(attn.to_q, "bias", None) is not None:
            specs[attn.to_q.bias] = ("model",)
            specs[attn.to_k.bias] = ("model",)
            specs[attn.to_v.bias] = ("model",)
        specs[attn.to_out[0].weight] = ("batch", "model")
        specs[attn.to_out[0].bias] = ("batch",)
        if getattr(attn.norm_q, "weight", None) is not None:
            specs[attn.norm_q.weight] = ("model",)
        if getattr(attn.norm_k, "weight", None) is not None:
            specs[attn.norm_k.weight] = ("model",)

        # FeedForward: net[0]=GELU(1920, 7680).proj, net[2]=Linear(7680, 1920)
        specs[block.ff.net[0].proj.weight] = ("model", "batch")
        specs[block.ff.net[0].proj.bias] = ("model",)
        specs[block.ff.net[2].weight] = ("batch", "model")
        specs[block.ff.net[2].bias] = ("batch",)

    # Final norm + AdaLN out + proj_out
    if getattr(transformer.norm_final, "weight", None) is not None:
        specs[transformer.norm_final.weight] = ("batch",)
    if getattr(transformer.norm_final, "bias", None) is not None:
        specs[transformer.norm_final.bias] = ("batch",)

    specs[transformer.norm_out.linear.weight] = ("model", "batch")
    specs[transformer.norm_out.linear.bias] = ("model",)
    if getattr(transformer.norm_out.norm, "weight", None) is not None:
        specs[transformer.norm_out.norm.weight] = ("batch",)
    if getattr(transformer.norm_out.norm, "bias", None) is not None:
        specs[transformer.norm_out.norm.bias] = ("batch",)

    # proj_out: Linear(1920 -> 64) — small output, shard input dim
    specs[transformer.proj_out.weight] = (None, "batch")
    specs[transformer.proj_out.bias] = (None,)

    return specs


# ---------------------------------------------------------------------------
# VAE decoder sharding
# ---------------------------------------------------------------------------


def shard_vae_decoder_specs(vae) -> dict:
    """Build shard specs for AutoencoderKLCogVideoX decoder weights.

    The VAE is memory-bound; we shard the first conv on "batch" to seed
    sharding through the decoder. CogVideoX-5b's VAE has no quant_conv /
    post_quant_conv (use_quant_conv=False).
    """
    specs = {
        vae.decoder.conv_in.conv.weight: ("batch", None, None, None, None),
        vae.decoder.conv_in.conv.bias: ("batch",),
    }
    if vae.post_quant_conv is not None:
        specs[vae.post_quant_conv.weight] = ("batch", None, None, None, None)
        specs[vae.post_quant_conv.bias] = ("batch",)
    return specs
