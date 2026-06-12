# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for Wan 2.2 A14B (14B) component tests.

A14B comes in two pipeline variants that share the same UMT5 text encoder
and Wan 2.1 VAE but use different transformer repos:

  - ``MODEL_ID_T2V`` — text-to-video (in_channels=16)
  - ``MODEL_ID_I2V`` — image-to-video (in_channels=36; 16 latent + 4 mask + 16 cond)

Both variants use a *dual* transformer (``transformer/`` high-noise expert +
``transformer_2/`` low-noise expert), gated by ``boundary_ratio`` at inference
time. ``load_dit(subfolder=...)`` picks one expert.

Exposes:
  - RESOLUTIONS: dict of 480p and 720p shape configs (Wan 2.1 VAE: scale=8)
  - load_umt5 / load_vae / load_dit / load_tokenizer: HuggingFace loaders
  - wan22_mesh: 2D ("batch", "model") SPMD mesh — adapts to device count
  - shard_*_specs: per-component shard spec functions
"""

import html
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import regex as re
import torch
import torch.nn as nn
from infra.utilities import Mesh
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from PIL import Image

from tests.infra.testers.compiler_config import CompilerConfig

MODEL_ID_T2V = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
MODEL_ID_I2V = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
# Default for standalone component tests (text encoder, VAE encoder/decoder,
# DiT in_channels=16). The e2e test selects per-mode.
MODEL_ID = MODEL_ID_T2V

# Boundary timestep ratio = boundary_ratio * num_train_timesteps. Above
# boundary → high-noise expert (``transformer``); below → low-noise expert
# (``transformer_2``).
BOUNDARY_RATIO = {"t2v": 0.875, "i2v": 0.9}

# Matches AutoencoderKLWan z_dim for the Wan 2.1 VAE that A14B uses.
# (5B's TI2V VAE is 48; A14B's is 16.)
LATENT_CHANNELS = 16
# Wan 2.1 VAE spatial scale factor (5B's is 16).
VAE_SCALE_FACTOR = 8
# VAE temporal stride. Same as 5B.
VAE_SCALE_FACTOR_TEMPORAL = 4

# Pixel and latent shapes per resolution. Latent dims are:
#   latent_frames = (num_frames - 1) // 4 + 1   (VAE temporal stride 4)
#   latent_h/w    = video_h/w // 8               (Wan 2.1 VAE spatial stride 8)
RESOLUTIONS = {
    "480p": {
        "video_h": 480,
        "video_w": 832,
        "num_frames": 81,
        "latent_frames": 21,
        "latent_h": 60,
        "latent_w": 104,
    },
    "720p": {
        "video_h": 720,
        "video_w": 1280,
        "num_frames": 81,
        "latent_frames": 21,
        "latent_h": 90,
        "latent_w": 160,
    },
}


# ---------------------------------------------------------------------------
# First-frame image loader (i2v conditioning input)
# ---------------------------------------------------------------------------


def load_first_frame_image(image_path: Path, height: int, width: int) -> torch.Tensor:
    """Load image at ``image_path``, scale-to-cover the target then center
    crop, and return a (1, 3, 1, H, W) bf16 tensor in [-1, 1] — the format
    the Wan VAE encoder expects for a single-frame image.

    Cover-style fit (vs. shorter-side fit) guarantees both target dims are
    reachable for any source aspect ratio.
    """
    img = Image.open(image_path).convert("RGB")
    src_w, src_h = img.size

    scale = max(width / src_w, height / src_h)
    new_w = max(width, round(src_w * scale))
    new_h = max(height, round(src_h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - width) // 2
    top = (new_h - height) // 2
    img = img.crop((left, top, left + width, top + height))

    arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    tensor = tensor * 2.0 - 1.0  # [-1, 1]
    return tensor.unsqueeze(0).unsqueeze(2).to(torch.bfloat16)  # (1, 3, 1, H, W)


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


def load_umt5(model_id: str = MODEL_ID):
    """Load UMT5-XXL text encoder with trained weights.

    T2V-A14B and I2V-A14B share the same encoder config and weights, so the
    default repo here is fine for either variant.
    """
    from transformers import UMT5EncoderModel

    enc = UMT5EncoderModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()

    # Checkpoint stores only ``shared.weight`` and relies on weight tying for
    # ``encoder.embed_tokens.weight``. Pinned transformers version doesn't
    # auto-tie on this subfolder load — every forward pass would otherwise
    # produce zero output. Fixed upstream in
    # https://github.com/huggingface/transformers/pull/43880.
    enc.encoder.embed_tokens.weight = enc.shared.weight

    return enc


def load_vae(model_id: str = MODEL_ID):
    """Load AutoencoderKLWan (Wan 2.1 VAE, z_dim=16) with trained weights."""
    from diffusers import AutoencoderKLWan

    return AutoencoderKLWan.from_pretrained(
        model_id,
        subfolder="vae",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval()


def load_dit(
    subfolder: str = "transformer",
    max_blocks: int = 0,
    model_id: str = MODEL_ID,
):
    """Load one WanTransformer3DModel expert with trained weights.

    A14B uses two experts:
      - ``subfolder="transformer"``   → high-noise expert
      - ``subfolder="transformer_2"`` → low-noise expert

    If ``max_blocks > 0``, truncate to that many transformer blocks
    (useful for compile-time debugging; full A14B has 40).
    """
    from diffusers import WanTransformer3DModel

    dit = WanTransformer3DModel.from_pretrained(
        model_id,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if max_blocks > 0 and len(dit.blocks) > max_blocks:
        dit.blocks = nn.ModuleList(list(dit.blocks[:max_blocks]))
    return dit.eval()


def load_tokenizer(model_id: str = MODEL_ID):
    """Load the UMT5 tokenizer used by Wan 2.2 A14B."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")


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
        if hasattr(sa, "relative_attention_bias"):
            specs[sa.relative_attention_bias.weight] = (None, "model")

        ffn = block.layer[1].DenseReluDense
        specs[ffn.wi_0.weight] = ("model", "batch")
        specs[ffn.wi_1.weight] = ("model", "batch")
        specs[ffn.wo.weight] = ("batch", "model")
        specs[block.layer[1].layer_norm.weight] = ("batch",)

    specs[encoder.encoder.final_layer_norm.weight] = ("batch",)
    return specs


# ---------------------------------------------------------------------------
# VAE sharding (Megatron column→row per WanResidualBlock)
# ---------------------------------------------------------------------------
#
# Inside every ``WanResidualBlock``:
#     norm1 → silu → conv1 → norm2 → silu → conv2 →+ shortcut
# the dual-conv pair is sharded Megatron-style: conv1 column-parallel
# (split C_out), conv2 row-parallel (split C_in). Two all-reduces fire per
# block: one inside ``norm2`` (the C-axis L2 sum, since the activation is
# C-sharded at that point) and one after ``conv2`` to sum the row-parallel
# partial outputs. ``norm2.gamma`` is sharded to match the C_out partition.
#
# Boundary convs / attention / upsamplers / quant_conv are left replicated.


def _pick_axis(mesh: Mesh) -> str:
    """Return the mesh axis with the most devices.

    For ``_MESH_SHAPES`` this is "batch" on Galaxy ``(8, 4)`` and "model"
    on every other device count.
    """
    sizes = dict(zip(mesh.axis_names, mesh.mesh_shape))
    return max(sizes, key=sizes.get)


def _megatron_pair_specs(block, axis: str) -> dict:
    """Per-WanResidualBlock Megatron column→row specs.

    conv1 column-parallel (shard C_out), norm2 gamma matched, conv2
    row-parallel (shard C_in). ``conv_shortcut``, ``norm1``, and
    ``conv2.bias`` stay replicated.
    """
    # Conv3d weight [C_out, C_in, kD, kH, kW]
    COL_W = (axis, None, None, None, None)  # column-parallel (shard C_out)
    ROW_W = (None, axis, None, None, None)  # row-parallel (shard C_in)
    COL_B = (axis,)  # column-parallel (shard bias)
    NORM_G = (axis, None, None, None)  # WanRMS_norm gamma [dim, 1, 1, 1]
    return {
        block.conv1.weight: COL_W,
        block.conv1.bias: COL_B,
        block.norm2.gamma: NORM_G,
        block.conv2.weight: ROW_W,
    }


def shard_vae_encoder_specs(vae, mesh: Mesh) -> dict:
    """Sharding specs for the AutoencoderKLWan encoder.

    Megatron column→row pair per block. conv1 col-parallel, norm2.gamma
    matched, conv2 row-parallel. 1 all_reduce per block (post-conv2 partial
    sum) + 1 internal all_reduce inside norm2 (the C-axis L2 sum).

    Left replicated: ``quant_conv``, ``encoder.conv_in`` (C_in=3),
    ``encoder.conv_out`` (C_out=32 = 2·z_dim), ``norm_out``, the
    ``mid_block`` attention (its fused ``to_qkv`` + chunk(3) pattern doesn't
    split cleanly on a 4-way mesh), and the ``WanResample`` downsamplers in
    ``down_blocks``.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanResidualBlock,
        WanResidualDownBlock
    )

    axis = _pick_axis(mesh)
    specs: dict = {}
    encoder = vae.encoder

    # down_blocks: flat ModuleList of WanResidualBlock / WanResample. Only
    # the WanResidualBlocks are sharded; downsamplers stay replicated.
    for m in encoder.down_blocks:
        if isinstance(m, WanResidualBlock):
            specs.update(_megatron_pair_specs(m, axis))
        elif isinstance(m, WanResidualDownBlock):
            for block in m.resnets:
                specs.update(_megatron_pair_specs(block, axis))

    # mid_block: 2 ResidualBlocks (attention stays replicated)
    for block in encoder.mid_block.resnets:
        specs.update(_megatron_pair_specs(block, axis))

    return specs


def shard_vae_decoder_specs(vae, mesh: Mesh) -> dict:
    """Sharding specs for the AutoencoderKLWan decoder.

    Megatron column→row pair per block. conv1 col-parallel, norm2.gamma matched,
    conv2 row-parallel. 1 all_reduce per block (post-conv2 partial sum) +
    1 internal all_reduce inside norm2 (the C-axis L2 sum).

    Left replicated: ``post_quant_conv``, ``decoder.conv_in`` (C_in=16),
    ``decoder.conv_out`` (C_out=3), ``norm_out``, the ``mid_block`` attention
    (its fused ``to_qkv`` + chunk(3) pattern doesn't split cleanly on a 4-way
    mesh), and the ``WanResample`` upsamplers in ``up_blocks``.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanResidualBlock

    axis = _pick_axis(mesh)
    specs: dict = {}
    decoder = vae.decoder

    # mid_block: WanResidualBlocks (attention stays replicated)
    for block in decoder.mid_block.resnets:
        specs.update(_megatron_pair_specs(block, axis))

    # 4 up_blocks: Upsamplers stay replicated.
    for up_block in decoder.up_blocks:
        for block in up_block.resnets:
            assert isinstance(block, WanResidualBlock)
            specs.update(_megatron_pair_specs(block, axis))

    return specs


# ---------------------------------------------------------------------------
# DiT sharding
# ---------------------------------------------------------------------------


# Tensor-parallel axis (Megatron column→row weight sharding); the other matrix
# dim and all boundary tensors stay replicated.
TP_AXIS = "model"
# Sequence-parallel axis for DiT activation sharding (the activation L dim has
# no analog in any weight, so it is sharded via constraints, not shard_dit_specs).
SP_AXIS = "batch"


def _megatron_linear_pair_specs(linear_1, linear_2) -> dict:
    """Megatron column→row shard specs for a 2-linear MLP.

    linear_1 is column-parallel (TP shards out-dim); linear_2 is row-parallel
    (TP shards in-dim) with bias replicated to be added on the TP-replicated
    activation produced by the post-matmul all-reduce.
    """
    return {
        linear_1.weight: (TP_AXIS, None),
        linear_1.bias: (TP_AXIS,),
        linear_2.weight: (None, TP_AXIS),
        linear_2.bias: (None,),
    }


def _shard_block_specs(block) -> dict:
    """Per-WanTransformerBlock shard specs.

    Applied to all of the 40 blocks. Three Megatron pairs
    (attn1, attn2, ffn) plus three norms with parameter sharding chosen to
    match the activation state at each point.
    """
    specs = {}

    # adaLN modulation parameter — replicated; broadcasts against STATE A.
    specs[block.scale_shift_table] = (None, None, None)

    # norm2 — affine LN on STATE A input; gamma/bias replicated to match.
    # When cross_attn_norm=False (not the A14B config), norm2 is nn.Identity
    # with no params, so guard the attribute access.
    if hasattr(block.norm2, "weight"):
        specs[block.norm2.weight] = (None,)
        specs[block.norm2.bias] = (None,)

    # Self and cross-attention: identical Megatron column→row pair.
    for attn in (block.attn1, block.attn2):
        # to_q / to_k / to_v — column-parallel (TP shards out-dim, = head split).
        specs[attn.to_q.weight] = (TP_AXIS, None)
        specs[attn.to_q.bias] = (TP_AXIS,)
        specs[attn.to_k.weight] = (TP_AXIS, None)
        specs[attn.to_k.bias] = (TP_AXIS,)
        specs[attn.to_v.weight] = (TP_AXIS, None)
        specs[attn.to_v.bias] = (TP_AXIS,)
        # norm_q / norm_k — gamma matches TP-sharded post-projection activation.
        specs[attn.norm_q.weight] = (TP_AXIS,)
        specs[attn.norm_k.weight] = (TP_AXIS,)
        # to_out — row-parallel (TP shards in-dim); AR after; bias replicated.
        specs[attn.to_out[0].weight] = (None, TP_AXIS)
        specs[attn.to_out[0].bias] = (None,)

    # FFN — Megatron col→row pair: net[0].proj (up D→4D), net[2] (down 4D→D).
    specs.update(_megatron_linear_pair_specs(block.ffn.net[0].proj, block.ffn.net[2]))

    return specs


def shard_dit_specs(dit) -> dict:
    """Build shard specs for WanTransformer3DModel weights.

    Boundary tensors (patch_embedding, proj_out, final scale_shift_table,
    time_proj) are replicated. Condition embedder uses Megatron column→row
    pairs for time_embedder and text_embedder.

    Works unchanged for T2V-A14B (in_channels=16) and I2V-A14B
    (in_channels=36): only the patch_embedding C_in differs, and it's
    replicated either way.
    """
    specs = {
        # Boundary tensors — replicated at block-stack entry and exit.
        dit.patch_embedding.weight: (None, None, None, None, None),
        dit.patch_embedding.bias: (None,),
        dit.proj_out.weight: (None, None),
        dit.proj_out.bias: (None,),
        # Final adaLN modulation parameter — replicated.
        dit.scale_shift_table: (None, None, None),
    }

    # Condition embedder — runs once per forward, feeds every block.
    ce = dit.condition_embedder
    # time_embedder : Megatron col→row pair (256 → D → D).
    specs.update(
        _megatron_linear_pair_specs(
            ce.time_embedder.linear_1, ce.time_embedder.linear_2
        )
    )
    # time_proj : D → 6·D, replicated (alone col-parallel without a row-parallel
    # partner would cost an AG before per-block adaLN broadcast).
    specs[ce.time_proj.weight] = (None, None)
    specs[ce.time_proj.bias] = (None,)
    # text_embedder : Megatron col→row pair (4096 → D → D).
    specs.update(
        _megatron_linear_pair_specs(
            ce.text_embedder.linear_1, ce.text_embedder.linear_2
        )
    )

    # 40 transformer blocks — identical sharding per block.
    for block in dit.blocks:
        specs.update(_shard_block_specs(block))

    return specs


def apply_dit_sp_activation_sharding(dit, mesh: Mesh) -> None:
    """Register forward hooks that introduce SP sharding on DiT activations.

    SP shards the activation L dim (no analog in any weight), so it has to be
    expressed via sharding_constraints on intermediates rather than via
    shard_dit_specs.

    When a constraint requests an L-shard immediately downstream of a reshape,
    Shardy back-propagates the shard through the reshape into one of the
    spatial dims that feed the merged L axis, which is something we don't want.
    We want the reshape to stay in replicated and let shard happen after the reshape.

    The fix is a back-to-back pair (replicated anchor, then L-sharded)
    downstream of the reshape: the first constraint terminates the
    back-propagation at a replicated state (reshape stays in replicated land
    and isn't partitioned), and the second introduces the shard between two
    non-reshape ops where Shardy inserts a clean scatter. The reverse
    direction (sharded → replicated upstream of a reshape, as at proj_out)
    doesn't trigger this bug - the AG lands before the reshape - so a single
    constraint suffices there.
    """
    from tt_torch.sharding import sharding_constraint_hook, sharding_constraint_tensor

    def _block_entry_pre_hook(module, args):
        hidden_states = args[0]
        hidden_states = sharding_constraint_tensor(
            hidden_states, mesh, (None, None, None)
        )
        hidden_states = sharding_constraint_tensor(
            hidden_states, mesh, (None, SP_AXIS, None)
        )
        return (hidden_states,) + args[1:]

    dit.blocks[0].register_forward_pre_hook(_block_entry_pre_hook)

    def _rope_hook(module, input, output):
        freqs_cos, freqs_sin = output
        freqs_cos = sharding_constraint_tensor(
            freqs_cos, mesh, (None, None, None, None)
        )
        freqs_sin = sharding_constraint_tensor(
            freqs_sin, mesh, (None, None, None, None)
        )
        freqs_cos = sharding_constraint_tensor(
            freqs_cos, mesh, (None, SP_AXIS, None, None)
        )
        freqs_sin = sharding_constraint_tensor(
            freqs_sin, mesh, (None, SP_AXIS, None, None)
        )
        return (freqs_cos, freqs_sin)

    dit.rope.register_forward_hook(_rope_hook)

    # proj_out output: force AG on L before the unpatchify reshape.
    proj_out_hook = sharding_constraint_hook(dit.proj_out, mesh, (None, None, None))
    dit.proj_out.register_forward_hook(proj_out_hook)


# ---------------------------------------------------------------------------
# Component runner (CPU or TT)
# ---------------------------------------------------------------------------


# Process-level cache: keep one ``torch.compile``-wrapped OptimizedModule per
# (model instance id, input-shape signature). Without this, every call to
# ``run_component`` builds a fresh OptimizedModule whose private dynamo cache
# is empty — so calling ``run_component(wrapper, ...)`` twice on the same
# wrapper triggers two full traces + two backend compiles. Strong refs on the
# wrappers keep ``id()`` stable for the cache's lifetime.
#
# Caller invariant: keep the wrapper alive across calls. If you ``del`` the
# wrapper and create a new one, ``id()`` may be reused for a *different*
# object — a silent cache hit returning the wrong compiled graph. For this
# test suite the ``_Components`` container in ``test_wan22_e2e.py`` keeps
# wrappers alive for the duration of each test.
_RUN_COMPONENT_COMPILE_CACHE: dict = {}


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
        enable_spmd()

    xr.set_device_type("TT")
    device = xm.xla_device()

    compiler_config = CompilerConfig(
        optimization_level=1,
        experimental_enable_dram_space_saving_optimization=True,
        enable_trace=True,
    )
    torch_xla.set_custom_compile_options(compiler_config.to_torch_compile_options())

    wrapper_on_device = wrapper.to(device)
    if hasattr(wrapper_on_device, "tie_weights"):
        wrapper_on_device.tie_weights()
    inputs_on_device = [t.to(device) for t in inputs]

    if use_sharding:
        assert shard_module is not None, "shard_fn requires shard_module"
        specs = shard_fn(shard_module)
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    shape_key = tuple((tuple(t.shape), t.dtype) for t in inputs_on_device)
    cache_key = (id(wrapper_on_device), shape_key)
    compiled = _RUN_COMPONENT_COMPILE_CACHE.get(cache_key)
    if compiled is None:
        compiled = torch.compile(wrapper_on_device, backend="tt")
        _RUN_COMPONENT_COMPILE_CACHE[cache_key] = compiled

    with torch.no_grad():
        out = compiled(*inputs_on_device)
    return out.to("cpu")


def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened and cast to float32."""
    x = x.detach().to("cpu").to(torch.float32).flatten()
    y = y.detach().to("cpu").to(torch.float32).flatten()
    vx, vy = x - x.mean(), y - y.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return float((vx @ vy) / denom)
