# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for Wan 2.2 TI2V-5B component tests.

Exposes:
  - RESOLUTIONS: dict of 480p and 720p shape configs
  - load_umt5 / load_vae / load_dit: real-weight HuggingFace model loaders
  - setup_mesh: 2D (batch, model) SPMD mesh (2/4/8/32 devices)
  - shard_umt5_weights / shard_vae_encoder_weights /
    shard_vae_decoder_weights / shard_dit_weights: per-component sharding
  - compare_cpu_tt: run model on CPU and TT, assert PCC > threshold
"""

import copy
import os
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

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


def _enable_spmd():
    """Enable SPMD mode. Must be called before creating a mesh."""
    import torch_xla.runtime as xr

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def setup_mesh():
    """Create a 2D ("batch", "model") SPMD mesh. Adapts to device count.

    Requires _enable_spmd() to have been called first.
    """
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh_shapes = {32: (8, 4), 8: (2, 4), 4: (1, 4), 2: (1, 2), 1: (1, 1)}
    if num_devices not in mesh_shapes:
        raise ValueError(
            f"Unsupported device count: {num_devices}. "
            f"Expected one of {sorted(mesh_shapes)}."
        )
    mesh = Mesh(device_ids, mesh_shapes[num_devices], ("batch", "model"))
    print(f"[SPMD] Mesh: {mesh_shapes[num_devices]} on {num_devices} devices")
    return mesh


# ---------------------------------------------------------------------------
# CPU vs TT comparison helper
# ---------------------------------------------------------------------------


def _compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson Correlation Coefficient between two tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()
    return torch.corrcoef(torch.stack([a_flat, b_flat]))[0, 1].item()


def _unwrap_output(out):
    """Unwrap diffusers BaseOutput objects and tuples to the primary tensor."""
    if hasattr(out, "sample"):  # diffusers BaseOutput
        return out.sample
    if hasattr(out, "last_hidden_state"):  # HF encoder output
        return out.last_hidden_state
    if isinstance(out, tuple):
        return out[0]
    return out


def _run_cpu(model: nn.Module, inputs: list) -> torch.Tensor:
    """Run model on CPU in isolation. No torch_xla imports happen here.

    Deep-copies the model so the caller's model is untouched (it'll be moved
    to TT by _run_tt afterwards).
    """
    cpu_model = copy.deepcopy(model).cpu().eval()
    cpu_inputs = [t.cpu() for t in inputs]

    with torch.no_grad():
        out = cpu_model(*cpu_inputs)
    out = _unwrap_output(out).float()

    del cpu_model  # free memory before TT run
    return out


def _run_tt(
    model: nn.Module, inputs: list, shard_fn: Optional[Callable] = None
) -> torch.Tensor:
    """Move model to TT, optionally shard, run forward, return output on CPU.

    This is the only place torch_xla gets imported / configured. Call AFTER
    _run_cpu has completed.
    """
    import torch_xla
    import torch_xla.runtime as xr

    xr.set_device_type("TT")
    if shard_fn is not None:
        _enable_spmd()
        mesh = setup_mesh()

    torch_xla.set_custom_compile_options({"optimization_level": 1})

    tt_device = torch_xla.device()
    # bfloat16() covers every param (Linear, Conv3d, LayerNorm...), then move.
    tt_model = model.eval().bfloat16().to(tt_device)

    tt_inputs = []
    for t in inputs:
        if t.is_floating_point():
            tt_inputs.append(t.bfloat16().to(tt_device))
        else:
            tt_inputs.append(t.to(tt_device))

    if shard_fn is not None:
        shard_fn(tt_model, mesh)

    with torch.no_grad():
        out = tt_model(*tt_inputs)
    out = _unwrap_output(out)
    return out.cpu().float()


def compare_cpu_tt(
    model: nn.Module,
    inputs: list,
    shard_fn: Optional[Callable] = None,
    required_pcc: float = 0.99,
) -> float:
    """Run model on CPU FIRST, then TT, then compare via PCC.

    Order matters: the CPU pass runs in complete isolation before any
    torch_xla import or device setup. This keeps the CPU baseline clean
    even when xla runtime state is preserved across tests.

    Args:
        model: nn.Module. Deep-copied for the CPU run; the original is moved
            to TT for the TT run.
        inputs: list of tensors; floating tensors are cast to bfloat16 on TT.
        shard_fn: optional (model_on_tt, mesh) -> None. When None, TT runs
            single-device (no SPMD).
        required_pcc: minimum PCC threshold.

    Returns:
        Computed PCC value.
    """
    # 1) CPU baseline — fully isolated, no torch_xla touched
    print("[CPU] Running baseline forward pass...")
    cpu_out = _run_cpu(model, inputs)
    print(
        f"[CPU] shape: {tuple(cpu_out.shape)}  "
        f"abs_mean: {cpu_out.abs().mean().item():.6f}  "
        f"abs_max: {cpu_out.abs().max().item():.6f}"
    )

    # 2) TT run — sharded or single-device
    mode = "sharded" if shard_fn is not None else "single-device"
    print(f"[TT]  Running {mode} forward pass...")
    tt_out = _run_tt(model, inputs, shard_fn)
    print(
        f"[TT]  shape: {tuple(tt_out.shape)}  "
        f"abs_mean: {tt_out.abs().mean().item():.6f}  "
        f"abs_max: {tt_out.abs().max().item():.6f}"
    )

    # 3) Compare
    pcc = _compute_pcc(tt_out, cpu_out)
    max_diff = (tt_out - cpu_out).abs().max().item()
    print(f"  PCC:      {pcc:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    assert pcc > required_pcc, f"PCC too low: {pcc:.6f} (required > {required_pcc})"
    return pcc


# ---------------------------------------------------------------------------
# UMT5 sharding
# ---------------------------------------------------------------------------


def shard_umt5_weights(encoder, mesh):
    """Apply 2D tensor-parallel sharding to UMT5EncoderModel weights.

    Column-parallel (q, k, v, wi_0, wi_1):  ("model", "batch")
    Row-parallel   (o, wo):                 ("batch", "model")
    """
    import torch_xla.distributed.spmd as xs

    # Embedding (vocab_size, d_model): shard d_model on "batch"
    xs.mark_sharding(encoder.shared.weight, mesh, (None, "batch"))

    for block in encoder.encoder.block:
        self_attn = block.layer[0].SelfAttention
        xs.mark_sharding(self_attn.q.weight, mesh, ("model", "batch"))
        xs.mark_sharding(self_attn.k.weight, mesh, ("model", "batch"))
        xs.mark_sharding(self_attn.v.weight, mesh, ("model", "batch"))
        xs.mark_sharding(self_attn.o.weight, mesh, ("batch", "model"))
        xs.mark_sharding(block.layer[0].layer_norm.weight, mesh, ("batch",))

        ffn = block.layer[1].DenseReluDense
        xs.mark_sharding(ffn.wi_0.weight, mesh, ("model", "batch"))
        xs.mark_sharding(ffn.wi_1.weight, mesh, ("model", "batch"))
        xs.mark_sharding(ffn.wo.weight, mesh, ("batch", "model"))
        xs.mark_sharding(block.layer[1].layer_norm.weight, mesh, ("batch",))

    xs.mark_sharding(encoder.encoder.final_layer_norm.weight, mesh, ("batch",))


# ---------------------------------------------------------------------------
# VAE encoder sharding
# ---------------------------------------------------------------------------


def shard_vae_encoder_weights(vae, mesh):
    """Apply sharding to AutoencoderKLWan encoder weights.

    The VAE is memory-bound, not compute-bound like the DiT. We shard only
    the largest conv outputs along "batch" to distribute parameters, leaving
    per-layer channel dims mostly replicated. quant_conv maps to latent
    (z_dim*2) channels — shard its output dim on "batch".
    """
    import torch_xla.distributed.spmd as xs

    # Quant conv (post-encoder, output = 2*z_dim): shard output channels
    xs.mark_sharding(vae.quant_conv.weight, mesh, ("batch", None, None, None, None))
    xs.mark_sharding(vae.quant_conv.bias, mesh, ("batch",))

    # Encoder: shard the first conv output channels to get sharding propagating
    # through the encoder via XLA's shape inference.
    xs.mark_sharding(
        vae.encoder.conv_in.weight, mesh, ("batch", None, None, None, None)
    )
    xs.mark_sharding(vae.encoder.conv_in.bias, mesh, ("batch",))


# ---------------------------------------------------------------------------
# VAE decoder sharding
# ---------------------------------------------------------------------------


def shard_vae_decoder_weights(vae, mesh):
    """Apply sharding to AutoencoderKLWan decoder weights.

    Mirrors the encoder strategy: shard post_quant_conv input and the
    decoder's first conv to seed sharding through the decoder.
    """
    import torch_xla.distributed.spmd as xs

    # Post-quant conv: input is z_dim latent, output = decoder base dim
    xs.mark_sharding(
        vae.post_quant_conv.weight, mesh, ("batch", None, None, None, None)
    )
    xs.mark_sharding(vae.post_quant_conv.bias, mesh, ("batch",))

    # Decoder first conv
    xs.mark_sharding(
        vae.decoder.conv_in.weight, mesh, ("batch", None, None, None, None)
    )
    xs.mark_sharding(vae.decoder.conv_in.bias, mesh, ("batch",))


# ---------------------------------------------------------------------------
# DiT sharding
# ---------------------------------------------------------------------------


def shard_dit_weights(dit, mesh):
    """Apply 2D tensor-parallel sharding to WanTransformer3DModel weights.

    Mesh axes: ("batch", "model")
    Column-parallel (QKV, FFN up):  ("model", "batch")
    Row-parallel   (O, FFN down):   ("batch", "model")
    """
    import torch_xla.distributed.spmd as xs

    # Patch embedding
    xs.mark_sharding(
        dit.patch_embedding.weight, mesh, ("batch", None, None, None, None)
    )
    xs.mark_sharding(dit.patch_embedding.bias, mesh, ("batch",))

    # Scale-shift table
    xs.mark_sharding(dit.scale_shift_table, mesh, (None, None, "batch"))

    # Condition embedder
    ce = dit.condition_embedder
    xs.mark_sharding(ce.time_embedder.linear_1.weight, mesh, ("model", "batch"))
    xs.mark_sharding(ce.time_embedder.linear_1.bias, mesh, ("model",))
    xs.mark_sharding(ce.time_embedder.linear_2.weight, mesh, ("batch", "model"))
    xs.mark_sharding(ce.time_embedder.linear_2.bias, mesh, ("batch",))
    xs.mark_sharding(ce.time_proj.weight, mesh, ("batch", None))
    xs.mark_sharding(ce.time_proj.bias, mesh, ("batch",))
    xs.mark_sharding(ce.text_embedder.linear_1.weight, mesh, ("model", "batch"))
    xs.mark_sharding(ce.text_embedder.linear_1.bias, mesh, ("model",))
    xs.mark_sharding(ce.text_embedder.linear_2.weight, mesh, ("batch", "model"))
    xs.mark_sharding(ce.text_embedder.linear_2.bias, mesh, ("batch",))

    # Output projection
    xs.mark_sharding(dit.proj_out.weight, mesh, (None, "batch"))
    xs.mark_sharding(dit.proj_out.bias, mesh, (None,))

    # Per-block sharding
    for block in dit.blocks:
        xs.mark_sharding(block.scale_shift_table, mesh, (None, None, "batch"))
        xs.mark_sharding(block.norm2.weight, mesh, ("batch",))
        xs.mark_sharding(block.norm2.bias, mesh, ("batch",))

        for attn in [block.attn1, block.attn2]:
            xs.mark_sharding(attn.to_q.weight, mesh, ("model", "batch"))
            xs.mark_sharding(attn.to_q.bias, mesh, ("model",))
            xs.mark_sharding(attn.to_k.weight, mesh, ("model", "batch"))
            xs.mark_sharding(attn.to_k.bias, mesh, ("model",))
            xs.mark_sharding(attn.to_v.weight, mesh, ("model", "batch"))
            xs.mark_sharding(attn.to_v.bias, mesh, ("model",))
            xs.mark_sharding(attn.to_out[0].weight, mesh, ("batch", "model"))
            xs.mark_sharding(attn.to_out[0].bias, mesh, ("batch",))
            xs.mark_sharding(attn.norm_q.weight, mesh, ("model",))
            xs.mark_sharding(attn.norm_k.weight, mesh, ("model",))

        xs.mark_sharding(block.ffn.net[0].proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(block.ffn.net[0].proj.bias, mesh, ("model",))
        xs.mark_sharding(block.ffn.net[2].weight, mesh, ("batch", "model"))
        xs.mark_sharding(block.ffn.net[2].bias, mesh, ("batch",))
