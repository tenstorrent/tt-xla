# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for Krea Realtime 14B component tests.

Exposes:
  - load_text_encoder / load_transformer / load_vae: HuggingFace model loaders
  - krea_mesh: 2D ("batch", "model") SPMD mesh — adapts to device count
  - shard_text_encoder_specs / shard_transformer_specs: per-component shard
    spec functions returning dict[Tensor, partition_spec] for run_graph_test.
"""

import torch
from diffusers import ModularPipeline
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh

# ---------------------------------------------------------------------------
# Model + dtype
# ---------------------------------------------------------------------------

REPO_ID = "krea/krea-realtime-video"
DTYPE = torch.bfloat16


# ---------------------------------------------------------------------------
# Default inference shape
# ---------------------------------------------------------------------------

HEIGHT = 480
WIDTH = 832
NUM_BLOCKS = 1
MAX_SEQ_LEN = 512  # text token length

LATENT_H = HEIGHT // 8  # 60
LATENT_W = WIDTH // 8  # 104
NUM_FRAMES_PER_BLOCK = 3
NUM_LATENT_FRAMES = NUM_BLOCKS * NUM_FRAMES_PER_BLOCK  # 3
NUM_CHANNELS_LATENTS = 16
TEXT_EMBED_DIM = 4096

# KV-cache config
KV_CACHE_NUM_FRAMES = 3
FRAME_SEQ_LENGTH = 1560
SEQ_LENGTH = 32760
LOCAL_ATTN_SIZE = KV_CACHE_NUM_FRAMES + NUM_FRAMES_PER_BLOCK  # 6
KV_CACHE_SIZE = LOCAL_ATTN_SIZE * FRAME_SEQ_LENGTH  # 9360


# ---------------------------------------------------------------------------
# Component loaders (real weights from HuggingFace)
# ---------------------------------------------------------------------------


def _load_pipe():
    """Load the full Krea ModularPipeline on CPU in bfloat16."""
    pipe = ModularPipeline.from_pretrained(REPO_ID, trust_remote_code=True)
    pipe.load_components(
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype={"default": DTYPE, "vae": DTYPE},
    )
    return pipe


def load_text_encoder():
    """Load the UMT5 text encoder."""
    return _load_pipe().text_encoder.eval()


def load_transformer():
    """Load the CausalWanModel transformer (the 14B DiT)."""
    return _load_pipe().transformer.eval()


def load_vae():
    """Load the 3D causal VAE."""
    return _load_pipe().vae.eval()


# ---------------------------------------------------------------------------
# SPMD mesh
# ---------------------------------------------------------------------------

# (batch, model) shapes by device count
_MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}


def krea_mesh() -> Mesh:
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


def shard_text_encoder_specs(encoder) -> dict:
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
# CausalWanModel (transformer) sharding
# ---------------------------------------------------------------------------


def shard_transformer_specs(transformer) -> dict:
    """Build shard specs for CausalWanModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (Q, K, V, FFN up):  ("model", "batch")
    Row-parallel   (O, FFN down):        ("batch", "model")

    Krea has self-attn + cross-attn per block (autoregressive video).
    """
    specs = {
        # Patch embedding (Conv3d 16 -> 5120)
        transformer.patch_embedding.weight: ("batch", None, None, None, None),
        transformer.patch_embedding.bias: ("batch",),
    }

    # text_embedding: Sequential[Linear(4096,5120), GELU, Linear(5120,5120)]
    specs[transformer.text_embedding[0].weight] = ("model", "batch")
    specs[transformer.text_embedding[0].bias] = ("model",)
    specs[transformer.text_embedding[2].weight] = ("batch", "model")
    specs[transformer.text_embedding[2].bias] = ("batch",)

    # time_embedding: Sequential[Linear(256,5120), SiLU, Linear(5120,5120)]
    specs[transformer.time_embedding[0].weight] = ("model", "batch")
    specs[transformer.time_embedding[0].bias] = ("model",)
    specs[transformer.time_embedding[2].weight] = ("batch", "model")
    specs[transformer.time_embedding[2].bias] = ("batch",)

    # time_projection: Sequential[SiLU, Linear(5120, 30720)]
    specs[transformer.time_projection[1].weight] = ("model", "batch")
    specs[transformer.time_projection[1].bias] = ("model",)

    # Per-block sharding
    for block in transformer.blocks:
        # Only norm3 has a learnable weight (elementwise_affine=True);
        # norm1 and norm2 use elementwise_affine=False.
        if hasattr(block.norm3, "weight") and block.norm3.weight is not None:
            specs[block.norm3.weight] = ("batch",)

        for attn in (block.self_attn, block.cross_attn):
            specs[attn.q.weight] = ("model", "batch")
            specs[attn.q.bias] = ("model",)
            specs[attn.k.weight] = ("model", "batch")
            specs[attn.k.bias] = ("model",)
            specs[attn.v.weight] = ("model", "batch")
            specs[attn.v.bias] = ("model",)
            specs[attn.o.weight] = ("batch", "model")
            specs[attn.o.bias] = ("batch",)
            if hasattr(attn.norm_q, "weight") and attn.norm_q.weight is not None:
                specs[attn.norm_q.weight] = ("model",)
            if hasattr(attn.norm_k, "weight") and attn.norm_k.weight is not None:
                specs[attn.norm_k.weight] = ("model",)

        # FFN: Sequential[Linear(5120,13824), GELU, Linear(13824,5120)]
        specs[block.ffn[0].weight] = ("model", "batch")
        specs[block.ffn[0].bias] = ("model",)
        specs[block.ffn[2].weight] = ("batch", "model")
        specs[block.ffn[2].bias] = ("batch",)

    # Head: Linear(5120 -> 64) - small output, shard input dim
    specs[transformer.head.head.weight] = (None, "batch")
    specs[transformer.head.head.bias] = (None,)

    return specs


# VAE is small enough to fit on a single chip — no sharding needed.
