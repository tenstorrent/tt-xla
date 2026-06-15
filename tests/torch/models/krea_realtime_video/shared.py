# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for krea/krea-realtime-video component tests.

krea-realtime-video is a real-time (CausVid) text-to-video DiT, a finetune of
Wan-AI/Wan2.1-T2V-14B. It is published as an HF *modular* DiffusionPipeline
(WanModularPipeline). Its components are brought up independently:

  - text_encoder : UMT5EncoderModel (UMT5-XXL, ~5.5B)   [Wan-AI/Wan2.1-T2V-14B-Diffusers]
  - vae          : AutoencoderKLWan (3D causal VAE)      [Wan-AI/Wan2.1-T2V-14B-Diffusers]
  - transformer  : CausalWanModel (~14.3B, custom code)  [krea/krea-realtime-video, trust_remote_code]

The text encoder and VAE are identical classes to the Wan 2.2 component tests
(tests/torch/models/wan2_2), so their loaders/shard specs mirror that file. The
transformer is the krea-specific causal model and is wrapped here to a
tensors-only forward (see KreaCausalDiTWrapper).

Exposes:
  - TRANSFORMER_CFG, RESOLUTIONS: shape constants captured from HF config
  - load_umt5 / load_vae / load_transformer: real-weight HuggingFace loaders
  - KreaCausalDiTWrapper: tensors-only wrapper that builds the per-block
    KV / cross-attention caches the causal model requires
  - krea_mesh: 2D ("batch", "model") SPMD mesh sized to device count
  - shard_umt5_specs / shard_vae_decoder_specs / shard_causal_dit_specs:
    per-component shard spec functions for run_graph_test
"""

import torch
import torch.nn as nn
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import get_mesh

# Transformer (custom causal code) lives in the krea repo; the encoder/VAE come
# from the Wan 2.1 diffusers mirror that the modular_model_index.json points at.
TRANSFORMER_ID = "krea/krea-realtime-video"
BASE_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

# CausalWanModel config (transformer/config.json) — drives input + cache shapes.
TRANSFORMER_CFG = {
    "dim": 5120,
    "ffn_dim": 13824,
    "num_heads": 40,
    "head_dim": 5120 // 40,  # 128
    "num_layers": 40,
    "in_dim": 16,
    "out_dim": 16,
    "text_len": 512,
    "text_dim": 4096,
    "patch_size": (1, 2, 2),
    "frame_seq_length": 1560,  # tokens per latent frame after patchify
    "num_frames_per_block": 3,  # latent frames denoised per realtime step
    "seq_length": 32760,  # full streaming cache size (local_attn_size == -1)
}

# Latent shapes per resolution. AutoencoderKLWan compresses 8x spatial / 4x
# temporal; the DiT patchifies with stride (1, 2, 2). For one realtime block of
# `num_frames_per_block` latent frames:
#   tokens/frame = (latent_h // 2) * (latent_w // 2) == frame_seq_length (1560)
RESOLUTIONS = {
    "480p": {
        "video_h": 480,
        "video_w": 832,
        "num_frames": 81,
        "latent_c": 16,
        "latent_frames": 21,  # (81 - 1) // 4 + 1
        "latent_h": 60,  # 480 // 8
        "latent_w": 104,  # 832 // 8
    },
}


# ---------------------------------------------------------------------------
# Model loaders (real weights from HuggingFace)
# ---------------------------------------------------------------------------


def load_umt5():
    """Load UMT5-XXL text encoder with trained weights."""
    from transformers import UMT5EncoderModel

    enc = UMT5EncoderModel.from_pretrained(
        BASE_ID,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()

    # The checkpoint stores only `shared.weight` and relies on weight-tying for
    # `encoder.embed_tokens.weight`. Our pinned transformers does not auto-tie
    # on this subfolder load, so tie it manually (mirrors wan2_2/shared.py).
    # Fixed upstream in huggingface/transformers#43880 — drop once we upgrade.
    enc.encoder.embed_tokens.weight = enc.shared.weight
    return enc


def load_vae():
    """Load AutoencoderKLWan (3D causal VAE) with trained weights."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        BASE_ID,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()


def _patch_cuda_device_assumptions(dit):
    """Loader-side workaround for a CUDA-only assumption in the krea code.

    transformer/model.py::sinusoidal_embedding_1d allocates with
    `device=torch.cuda.current_device()`, which raises on CPU and on the TT
    backend ("Torch not compiled with CUDA enabled"). Replace it with a
    device-agnostic version on the model's dynamically-loaded (trust_remote_code)
    module — a monkey-patch, never an edit to the cached source tree.

    The other torch.cuda references (attention.py) are already guarded by
    `torch.cuda.is_available()` and fall back to scaled_dot_product_attention.
    """
    import sys

    mod = sys.modules[type(dit).__module__]

    def sinusoidal_embedding_1d(dim, position):
        assert dim % 2 == 0
        half = dim // 2
        position = position.type(torch.float64)
        sinusoid = torch.outer(
            position,
            torch.pow(
                10000,
                -torch.arange(half, device=position.device, dtype=torch.float64).div(
                    half
                ),
            ),
        )
        return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)

    mod.sinusoidal_embedding_1d = sinusoidal_embedding_1d
    return dit


def load_transformer(max_blocks: int = 0):
    """Load CausalWanModel (the krea causal DiT) with trained weights.

    Loaded via AutoModel + trust_remote_code because the model class lives in
    the repo (transformer/causal_model.py, auto_map -> CausalWanModel).

    If max_blocks > 0, truncate to that many transformer blocks (compile-time
    debugging; the full model has 40).
    """
    from diffusers import AutoModel

    dit = AutoModel.from_pretrained(
        TRANSFORMER_ID,
        subfolder="transformer",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    _patch_cuda_device_assumptions(dit)
    if max_blocks > 0 and len(dit.blocks) > max_blocks:
        dit.blocks = nn.ModuleList(list(dit.blocks[:max_blocks]))
    return dit.eval()


# ---------------------------------------------------------------------------
# Causal DiT wrapper — builds the KV / cross-attn caches and adapts list I/O
# ---------------------------------------------------------------------------


def _new_caches(num_blocks, batch, kv_len, num_heads, head_dim, dtype):
    """Allocate fresh per-block KV and cross-attention caches.

    Mirrors before_denoise.py::_initialize_kv_cache / _initialize_crossattn_cache.
    `kv_len` is the streaming cache length. For a single realtime step starting
    at frame 0 only the first (num_frames_per_block * frame_seq_length) entries
    are written/read, so the test sizes the cache to exactly one block to keep
    memory tractable (the full pipeline uses seq_length == 32760).
    """
    kv_cache = []
    crossattn_cache = []
    for _ in range(num_blocks):
        kv_cache.append(
            {
                "k": torch.zeros(
                    batch, kv_len, num_heads, head_dim, dtype=dtype
                ).contiguous(),
                "v": torch.zeros(
                    batch, kv_len, num_heads, head_dim, dtype=dtype
                ).contiguous(),
                "global_end_index": 0,
                "local_end_index": 0,
            }
        )
        crossattn_cache.append(
            {
                "k": torch.zeros(
                    batch, 512, num_heads, head_dim, dtype=dtype
                ).contiguous(),
                "v": torch.zeros(
                    batch, 512, num_heads, head_dim, dtype=dtype
                ).contiguous(),
                "is_init": False,
            }
        )
    return kv_cache, crossattn_cache


class KreaCausalDiTWrapper(torch.nn.Module):
    """Tensors-only forward over CausalWanModel for one realtime denoise step.

    The underlying model takes lists of tensors plus mutable per-block caches
    (kv_cache, crossattn_cache) and structural ints (seq_len, current_start).
    This wrapper pins those: it accepts a single batched latent tensor, a
    timestep tensor, and a single context tensor, rebuilds the caches fresh on
    every forward, and returns the denoised latent as a bare tensor — so
    run_graph_test traces a clean graph and needs no unpack_forward_output.
    """

    def __init__(self, dit, kv_cache_frames: int = None):
        super().__init__()
        self.dit = dit
        cfg = TRANSFORMER_CFG
        self.num_heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.seq_len = cfg["seq_length"]
        # Cache only what one realtime block touches (frame 0 step).
        frames = kv_cache_frames or cfg["num_frames_per_block"]
        self.kv_len = frames * cfg["frame_seq_length"]

    def forward(self, latents, timestep, context):
        num_blocks = len(self.dit.blocks)
        kv_cache, crossattn_cache = _new_caches(
            num_blocks,
            latents.shape[0],
            self.kv_len,
            self.num_heads,
            self.head_dim,
            latents.dtype,
        )
        out = self.dit(
            x=[latents[i] for i in range(latents.shape[0])],
            t=timestep,
            context=[context[i] for i in range(context.shape[0])],
            kv_cache=kv_cache,
            seq_len=self.seq_len,
            crossattn_cache=crossattn_cache,
            current_start=0,
            cache_start=None,
        )
        return out


# ---------------------------------------------------------------------------
# SPMD mesh
# ---------------------------------------------------------------------------

_MESH_SHAPES = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}


def krea_mesh() -> Mesh:
    """2D ("batch", "model") SPMD mesh sized to current device count."""
    import torch_xla.runtime as xr

    n = xr.global_runtime_device_count()
    if n not in _MESH_SHAPES:
        raise ValueError(
            f"Unsupported device count: {n}. Expected one of {sorted(_MESH_SHAPES)}."
        )
    return get_mesh(_MESH_SHAPES[n], ("batch", "model"))


# ---------------------------------------------------------------------------
# UMT5 sharding  (identical layout to wan2_2/shared.py — same UMT5-XXL class)
# ---------------------------------------------------------------------------


def shard_umt5_specs(encoder) -> dict:
    """Build shard specs for UMT5EncoderModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (q, k, v, wi_0, wi_1): ("model", "batch")
    Row-parallel   (o, wo):                ("batch", "model")
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
# VAE decoder sharding (AutoencoderKLWan — same class as wan2_2)
# ---------------------------------------------------------------------------


def shard_vae_decoder_specs(vae) -> dict:
    """Shard post_quant_conv input and the decoder's first conv (memory-bound)."""
    return {
        vae.post_quant_conv.weight: ("batch", None, None, None, None),
        vae.post_quant_conv.bias: ("batch",),
        vae.decoder.conv_in.weight: ("batch", None, None, None, None),
        vae.decoder.conv_in.bias: ("batch",),
    }


# ---------------------------------------------------------------------------
# Causal DiT sharding (CausalWanModel)
# ---------------------------------------------------------------------------


def shard_causal_dit_specs(dit) -> dict:
    """Build shard specs for CausalWanModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (QKV, FFN up):  ("model", "batch")
    Row-parallel   (O, FFN down):   ("batch", "model")

    CausalWanModel uses self_attn/cross_attn with .q/.k/.v/.o Linear layers and
    an nn.Sequential ffn (ffn[0] up, ffn[2] down) — different attribute names
    from diffusers' WanTransformer3DModel, but the same column/row strategy.
    """
    specs = {
        # Patch embedding (Conv3d in_dim -> dim)
        dit.patch_embedding.weight: ("batch", None, None, None, None),
        dit.patch_embedding.bias: ("batch",),
    }

    # text_embedding / time_embedding / time_projection are small nn.Sequentials
    specs[dit.text_embedding[0].weight] = ("model", "batch")
    specs[dit.text_embedding[0].bias] = ("model",)
    specs[dit.text_embedding[2].weight] = ("batch", "model")
    specs[dit.text_embedding[2].bias] = ("batch",)
    specs[dit.time_embedding[0].weight] = ("model", "batch")
    specs[dit.time_embedding[0].bias] = ("model",)
    specs[dit.time_embedding[2].weight] = ("batch", "model")
    specs[dit.time_embedding[2].bias] = ("batch",)
    specs[dit.time_projection[1].weight] = ("batch", "model")
    specs[dit.time_projection[1].bias] = ("batch",)

    for block in dit.blocks:
        for attn in [block.self_attn, block.cross_attn]:
            specs[attn.q.weight] = ("model", "batch")
            specs[attn.q.bias] = ("model",)
            specs[attn.k.weight] = ("model", "batch")
            specs[attn.k.bias] = ("model",)
            specs[attn.v.weight] = ("model", "batch")
            specs[attn.v.bias] = ("model",)
            specs[attn.o.weight] = ("batch", "model")
            specs[attn.o.bias] = ("batch",)

        # FFN: Sequential(Linear up, GELU, Linear down)
        specs[block.ffn[0].weight] = ("model", "batch")
        specs[block.ffn[0].bias] = ("model",)
        specs[block.ffn[2].weight] = ("batch", "model")
        specs[block.ffn[2].bias] = ("batch",)

    # Output head
    specs[dit.head.head.weight] = (None, "batch")
    specs[dit.head.head.bias] = (None,)
    return specs
