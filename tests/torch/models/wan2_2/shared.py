# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for Wan 2.2 TI2V-5B component tests.

Exposes:
  - RESOLUTIONS: dict of 480p and 720p shape configs
  - load_umt5 / load_vae / load_dit: real-weight HuggingFace model loaders
  - wan22_mesh: 2D ("batch", "model") SPMD mesh — adapts to device count
  - shard_umt5_specs / shard_vae_encoder_specs /
    shard_vae_decoder_specs / shard_dit_specs: per-component shard spec
    functions returning dict[Tensor, partition_spec] for run_graph_test.
"""

import torch
import torch.nn as nn
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Pixel and latent shapes per resolution. Latent dims are:
#   latent_frames = (num_frames - 1) // 4 + 1   (VAE temporal stride 4)
#   latent_h/w    = video_h/w // 16              (VAE spatial stride 16)
RESOLUTIONS = {
    "480p": {
        "video_h": 480,
        "video_w": 832,
        "num_frames": 81,
        "latent_frames": 21,
        "latent_h": 30,
        "latent_w": 52,
    },
    "720p": {
        "video_h": 704,
        "video_w": 1280,
        "num_frames": 121,
        "latent_frames": 31,
        "latent_h": 44,
        "latent_w": 80,
    },
}


# ---------------------------------------------------------------------------
# Model loaders (real weights from HuggingFace)
# ---------------------------------------------------------------------------


def load_umt5():
    """Load UMT5-XXL text encoder with trained weights."""
    from transformers import UMT5EncoderModel

    enc = UMT5EncoderModel.from_pretrained(
        MODEL_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()

    # The checkpoint stores only `shared.weight` and relies on weight-tying
    # for `encoder.embed_tokens.weight`. Our pinned transformers version does
    # not auto-tie on this subfolder load, so `encoder.embed_tokens.weight`
    # stays at its zero init and every forward pass produces zero output.
    # Fixed upstream in https://github.com/huggingface/transformers/pull/43880 -
    # remove this manual tie once we upgrade past that PR.
    enc.encoder.embed_tokens.weight = enc.shared.weight

    return enc


def load_vae():
    """Load AutoencoderKLWan (3D Causal VAE) with trained weights."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()


def load_dit(max_blocks: int = 0):
    """Load WanTransformer3DModel (DiT) with trained weights.

    If max_blocks > 0, truncate to that many transformer blocks
    (useful for compile-time debugging; full model has 30).
    """
    from diffusers import WanTransformer3DModel

    dit = WanTransformer3DModel.from_pretrained(
        MODEL_ID,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if max_blocks > 0 and len(dit.blocks) > max_blocks:
        dit.blocks = nn.ModuleList(list(dit.blocks[:max_blocks]))
    return dit.eval()


# ---------------------------------------------------------------------------
# SPMD mesh
# ---------------------------------------------------------------------------

_MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}


def wan22_mesh() -> Mesh:
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
# UMT5 sharding
# ---------------------------------------------------------------------------


def shard_umt5_specs(encoder) -> dict:
    """Build shard specs for UMT5EncoderModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, wi_0, wi_1):  ("model", "batch")
    Row-parallel   (o, wo):                 ("batch", "model")
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
        specs[ffn.wi_0.weight] = ("model", "batch")
        specs[ffn.wi_1.weight] = ("model", "batch")
        specs[ffn.wo.weight] = ("batch", "model")
        specs[block.layer[1].layer_norm.weight] = ("batch",)

    specs[encoder.encoder.final_layer_norm.weight] = ("batch",)
    return specs


# ---------------------------------------------------------------------------
# VAE encoder sharding
# ---------------------------------------------------------------------------


def shard_vae_encoder_specs(vae) -> dict:
    """Build shard specs for AutoencoderKLWan encoder weights.

    The VAE is memory-bound, not compute-bound like the DiT. We shard only
    the largest conv outputs along "batch" to distribute parameters, leaving
    per-layer channel dims mostly replicated. quant_conv maps to latent
    (z_dim*2) channels — shard its output dim on "batch".
    """
    return {
        vae.quant_conv.weight: ("batch", None, None, None, None),
        vae.quant_conv.bias: ("batch",),
        vae.encoder.conv_in.weight: ("batch", None, None, None, None),
        vae.encoder.conv_in.bias: ("batch",),
    }


# ---------------------------------------------------------------------------
# VAE decoder sharding
# ---------------------------------------------------------------------------


def shard_vae_decoder_specs(vae) -> dict:
    """Build shard specs for AutoencoderKLWan decoder weights.

    Mirrors the encoder strategy: shard post_quant_conv input and the
    decoder's first conv to seed sharding through the decoder.
    """
    return {
        vae.post_quant_conv.weight: ("batch", None, None, None, None),
        vae.post_quant_conv.bias: ("batch",),
        vae.decoder.conv_in.weight: ("batch", None, None, None, None),
        vae.decoder.conv_in.bias: ("batch",),
    }


# ---------------------------------------------------------------------------
# DiT sharding
# ---------------------------------------------------------------------------


def shard_dit_specs(dit) -> dict:
    """Build shard specs for WanTransformer3DModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (QKV, FFN up):  ("model", "batch")
    Row-parallel   (O, FFN down):   ("batch", "model")
    """
    specs = {
        # Patch embedding
        dit.patch_embedding.weight: ("batch", None, None, None, None),
        dit.patch_embedding.bias: ("batch",),
        # Scale-shift table
        dit.scale_shift_table: (None, None, "batch"),
        # Output projection
        dit.proj_out.weight: (None, "batch"),
        dit.proj_out.bias: (None,),
    }

    # Condition embedder
    ce = dit.condition_embedder
    specs[ce.time_embedder.linear_1.weight] = ("model", "batch")
    specs[ce.time_embedder.linear_1.bias] = ("model",)
    specs[ce.time_embedder.linear_2.weight] = ("batch", "model")
    specs[ce.time_embedder.linear_2.bias] = ("batch",)
    specs[ce.time_proj.weight] = ("batch", None)
    specs[ce.time_proj.bias] = ("batch",)
    specs[ce.text_embedder.linear_1.weight] = ("model", "batch")
    specs[ce.text_embedder.linear_1.bias] = ("model",)
    specs[ce.text_embedder.linear_2.weight] = ("batch", "model")
    specs[ce.text_embedder.linear_2.bias] = ("batch",)

    # Per-block sharding
    for block in dit.blocks:
        specs[block.scale_shift_table] = (None, None, "batch")
        specs[block.norm2.weight] = ("batch",)
        specs[block.norm2.bias] = ("batch",)

        for attn in [block.attn1, block.attn2]:
            specs[attn.to_q.weight] = ("model", "batch")
            specs[attn.to_q.bias] = ("model",)
            specs[attn.to_k.weight] = ("model", "batch")
            specs[attn.to_k.bias] = ("model",)
            specs[attn.to_v.weight] = ("model", "batch")
            specs[attn.to_v.bias] = ("model",)
            specs[attn.to_out[0].weight] = ("batch", "model")
            specs[attn.to_out[0].bias] = ("batch",)
            specs[attn.norm_q.weight] = ("model",)
            specs[attn.norm_k.weight] = ("model",)

        specs[block.ffn.net[0].proj.weight] = ("model", "batch")
        specs[block.ffn.net[0].proj.bias] = ("model",)
        specs[block.ffn.net[2].weight] = ("batch", "model")
        specs[block.ffn.net[2].bias] = ("batch",)

    return specs
