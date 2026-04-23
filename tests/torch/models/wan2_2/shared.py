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

import html
from typing import Callable, Optional

import regex as re
import torch
import torch.nn as nn
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Matches WanTransformer3DModel.config.in_channels for TI2V-5B
# (= VAE z_dim = 48). Named here so the denoise loop avoids a magic number.
LATENT_CHANNELS = 48
# Wan 2.2 TI2V-5B VAE scale factor
VAE_SCALE_FACTOR = 16

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
# Prompt cleaning — matches diffusers/pipeline_wan.py:78-93 and Wan repo.
# ---------------------------------------------------------------------------


def _basic_clean(text: str) -> str:
    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    return html.unescape(html.unescape(text)).strip()


def _whitespace_clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prompt_clean(text: str) -> str:
    """Normalize prompt text the same way diffusers.WanPipeline does."""
    return _whitespace_clean(_basic_clean(text))


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


def load_tokenizer():
    """Load the UMT5 tokenizer used by Wan 2.2 TI2V-5B."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")


# ---------------------------------------------------------------------------
# Component wrappers (reused by component tests and the e2e test)
# ---------------------------------------------------------------------------


class UMT5Wrapper(nn.Module):
    """Return last_hidden_state as a plain tensor (not a model output object)."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class VAEEncoderWrapper(nn.Module):
    """Run encoder and return the deterministic mean latent."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(x).latent_dist.mean


class VAEDecoderWrapper(nn.Module):
    """Run decoder and return the reconstructed sample tensor."""

    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


class WanDiTWrapper(nn.Module):
    """Return the velocity tensor from the diffusers output tuple."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, timestep, encoder_hidden_states):
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


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


# ---------------------------------------------------------------------------
# Component runner (CPU or TT)
# ---------------------------------------------------------------------------


def run_component(
    wrapper: nn.Module,
    inputs: list,
    on_tt: bool,
    *,
    mesh: Optional[Mesh] = None,
    shard_module: Optional[nn.Module] = None,
    shard_fn: Optional[Callable[[nn.Module], dict]] = None,
) -> torch.Tensor:
    """Run a component wrapper on CPU or TT and return the CPU output tensor.

    CPU path (`on_tt=False`): calls ``wrapper(*inputs)`` under ``torch.no_grad()``.

    TT path (`on_tt=True`): moves ``wrapper`` and ``inputs`` onto the XLA
    device, optionally applies ``xs.mark_sharding`` using
    ``shard_fn(shard_module)`` and ``mesh``, then ``torch.compile(..., backend="tt")``
    and runs. Returns ``.to("cpu")`` of the result.

    Sharding is applied iff ``shard_fn`` is provided and ``mesh`` has more than
    one device. For a single-device mesh the shard branch is a silent no-op so
    the same call site works on 1-device and N-device setups. When
    ``shard_fn`` is provided, ``shard_module`` is required — it is the
    submodule whose device-resident parameters get sharded (e.g. ``wrapper.dit``).
    """
    if not on_tt:
        with torch.no_grad():
            return wrapper(*inputs)

    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

    use_sharding = (
        shard_fn is not None and mesh is not None and len(mesh.device_ids) > 1
    )

    if use_sharding:
        # Must run before any XLA op: sets CONVERT_SHLO_TO_SHARDY=1 and
        # xr.use_spmd(). Both are idempotent across calls.
        enable_spmd()

    xr.set_device_type("TT")
    device = xm.xla_device()

    torch_xla.set_custom_compile_options({"optimization_level": 1})

    # Move the raw wrapper and inputs to device first. shard_fn needs to see
    # XLA tensors, so sharding must happen *after* this move.
    wrapper_on_device = wrapper.to(device)
    if hasattr(wrapper_on_device, "tie_weights"):
        wrapper_on_device.tie_weights()
    inputs_on_device = [t.to(device) for t in inputs]

    if use_sharding:
        assert shard_module is not None, "shard_fn requires shard_module"
        specs = shard_fn(shard_module)
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    # Compile after the move + annotations so torch.compile traces the
    # on-device, already-sharded module on its first call.
    compiled = torch.compile(wrapper_on_device, backend="tt")

    with torch.no_grad():
        out = compiled(*inputs_on_device)
    return out.to("cpu")
