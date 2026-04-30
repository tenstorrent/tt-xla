# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for HunyuanVideo 1.5 (480p t2v distilled) component tests.

Exposes:
  - load_text_encoder / load_text_encoder_2 / load_transformer / load_vae:
    HuggingFace model loaders
  - hunyuan_mesh: 2D ("batch", "model") SPMD mesh — adapts to device count
  - shard_text_encoder_specs / shard_transformer_specs:
    per-component shard spec functions for run_graph_test.
"""

import torch
from diffusers import HunyuanVideo15Pipeline
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh

# ---------------------------------------------------------------------------
# Model + dtype
# ---------------------------------------------------------------------------

REPO_ID = "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Default inference shape
# ---------------------------------------------------------------------------

NUM_FRAMES = 17  # smallest valid (4k+1) — for smoke
NUM_LATENT_FRAMES = 5  # (17-1)//4 + 1
LATENT_H = 30  # 480 // 16
LATENT_W = 53  # 848 // 16

# Channel configs
NUM_CHANNELS_LATENTS = 32  # VAE latent channels
TRANSFORMER_IN_CHANNELS = 65  # 32 latent + 32 cond + 1 mask (concat)

# Text encoder dims
TEXT_EMBED_DIM = 3584  # Qwen2.5-VL hidden_state dim
TEXT_EMBED_2_DIM = 1472  # ByT5 hidden_state dim

# Image embeds (zeros for t2v; pipeline builds these and passes to transformer)
IMAGE_EMBED_DIM = 1152  # transformer.config.image_embed_dim
IMAGE_EMBED_SEQ = 64  # pipeline.vision_num_semantic_tokens default

# Token sequence lengths (from default prompt)
TEXT_TOKEN_MAX_LEN = 1108  # Qwen2.5-VL tokenized prompt length
TEXT_TOKEN_2_MAX_LEN = 256  # ByT5 tokenizer_2_max_length default
TRANSFORMER_TEXT_SEQ = 1000  # encoder_hidden_states seq dim into transformer
TRANSFORMER_TEXT_2_SEQ = 256  # encoder_hidden_states_2 seq dim


# ---------------------------------------------------------------------------
# Component loaders (real weights from HuggingFace)
# ---------------------------------------------------------------------------


def _load_pipe():
    """Load the full HunyuanVideo 1.5 pipeline on CPU in bfloat16."""
    return HunyuanVideo15Pipeline.from_pretrained(REPO_ID, torch_dtype=DTYPE)


def load_text_encoder():
    """Load the Qwen2.5-VL text encoder (text_encoder)."""
    return _load_pipe().text_encoder.eval()


def load_text_encoder_2():
    """Load the ByT5 text encoder (text_encoder_2)."""
    return _load_pipe().text_encoder_2.eval()


def load_transformer():
    """Load the HunyuanVideo15Transformer3DModel (DiT)."""
    return _load_pipe().transformer.eval()


def load_vae():
    """Load the AutoencoderKLHunyuanVideo15 VAE."""
    return _load_pipe().vae.eval()


# ---------------------------------------------------------------------------
# SPMD mesh
# ---------------------------------------------------------------------------

# (batch, model) shapes by device count
_MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}


def hunyuan_mesh() -> Mesh:
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
# Qwen2.5-VL (text_encoder) sharding
# ---------------------------------------------------------------------------


def shard_text_encoder_specs(encoder) -> dict:
    """Build shard specs for Qwen2.5-VL text encoder weights.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, gate, up):  ("model", "batch")
    Row-parallel   (o, down):              ("batch", "model")
    """
    specs = {}

    # Embedding (vocab, hidden_size)
    if hasattr(encoder, "embed_tokens"):
        specs[encoder.embed_tokens.weight] = (None, "batch")

    layers = getattr(encoder, "layers", None)
    if layers is None:
        return specs

    for layer in layers:
        # Self-attention
        sa = layer.self_attn
        specs[sa.q_proj.weight] = ("model", "batch")
        if sa.q_proj.bias is not None:
            specs[sa.q_proj.bias] = ("model",)
        specs[sa.k_proj.weight] = ("model", "batch")
        if sa.k_proj.bias is not None:
            specs[sa.k_proj.bias] = ("model",)
        specs[sa.v_proj.weight] = ("model", "batch")
        if sa.v_proj.bias is not None:
            specs[sa.v_proj.bias] = ("model",)
        specs[sa.o_proj.weight] = ("batch", "model")

        # MLP (gate / up / down — Llama/Qwen-style)
        mlp = layer.mlp
        specs[mlp.gate_proj.weight] = ("model", "batch")
        specs[mlp.up_proj.weight] = ("model", "batch")
        specs[mlp.down_proj.weight] = ("batch", "model")

        # Norms (replicated; small)
        specs[layer.input_layernorm.weight] = ("batch",)
        specs[layer.post_attention_layernorm.weight] = ("batch",)

    if hasattr(encoder, "norm"):
        specs[encoder.norm.weight] = ("batch",)

    return specs


# text_encoder_2 (ByT5, 0.22B params) is small enough to fit on a single chip — no sharding needed.


# ---------------------------------------------------------------------------
# HunyuanVideo15Transformer3DModel sharding
# ---------------------------------------------------------------------------


def shard_transformer_specs(transformer) -> dict:
    """Build shard specs for HunyuanVideo15Transformer3DModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (Q, K, V, FFN up):  ("model", "batch")
    Row-parallel   (O, FFN down):        ("batch", "model")
    """
    specs = {}

    # Patch embedding (Conv3d -> inner_dim)
    if hasattr(transformer.x_embedder, "proj"):
        specs[transformer.x_embedder.proj.weight] = ("batch", None, None, None, None)
        if transformer.x_embedder.proj.bias is not None:
            specs[transformer.x_embedder.proj.bias] = ("batch",)

    # Per-block sharding (dual-stream blocks)
    for block in transformer.transformer_blocks:
        # AdaLayerNormZero modulation projections (hidden -> 6*hidden)
        for norm_name in ("norm1", "norm1_context"):
            if hasattr(block, norm_name) and hasattr(
                getattr(block, norm_name), "linear"
            ):
                lin = getattr(block, norm_name).linear
                specs[lin.weight] = ("model", "batch")
                if lin.bias is not None:
                    specs[lin.bias] = ("model",)

        # Joint attention projections
        if hasattr(block, "attn"):
            attn = block.attn
            for proj_name in (
                "to_q",
                "to_k",
                "to_v",
                "add_q_proj",
                "add_k_proj",
                "add_v_proj",
            ):
                if hasattr(attn, proj_name):
                    proj = getattr(attn, proj_name)
                    specs[proj.weight] = ("model", "batch")
                    if proj.bias is not None:
                        specs[proj.bias] = ("model",)
            for proj_name in ("to_out", "to_add_out"):
                if hasattr(attn, proj_name):
                    out = getattr(attn, proj_name)
                    # to_out is typically nn.ModuleList([Linear, Dropout]) in diffusers Attention
                    target = (
                        out[0]
                        if isinstance(out, (torch.nn.Sequential, torch.nn.ModuleList))
                        else out
                    )
                    specs[target.weight] = ("batch", "model")
                    if target.bias is not None:
                        specs[target.bias] = ("batch",)

        # FeedForward (img + context streams)
        for ff_name in ("ff", "ff_context"):
            if hasattr(block, ff_name):
                ff = getattr(block, ff_name)
                # net[0] is activation wrapper (GELU/GEGLU/ApproximateGELU) — has .proj Linear
                if hasattr(ff.net[0], "proj"):
                    specs[ff.net[0].proj.weight] = ("model", "batch")
                    if ff.net[0].proj.bias is not None:
                        specs[ff.net[0].proj.bias] = ("model",)
                # net[2] is output Linear
                specs[ff.net[2].weight] = ("batch", "model")
                if ff.net[2].bias is not None:
                    specs[ff.net[2].bias] = ("batch",)

    # Output projection
    specs[transformer.proj_out.weight] = (None, "batch")
    if transformer.proj_out.bias is not None:
        specs[transformer.proj_out.bias] = (None,)

    return specs


# VAE runs single-device (1.26B params fits); current TT-XLA hits a hang during decode — to debug.
