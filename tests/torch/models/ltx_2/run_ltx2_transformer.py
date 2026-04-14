# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Dual-Stream DiT Transformer — standalone bringup script.

Component: LTX2VideoTransformer3DModel

Replicated mode: Minimal 4-layer config with random weights, single device.
  - Validates the dual-stream attention path (video + audio tokens).
  - Same dimensions as full model: video 32 heads x 128 dim, audio 32 heads x 64 dim.

TP mode (--tp): 2-layer config with 4-way tensor-parallel sharding (Megatron-style).
  - Enables SPMD with Shardy, creates mesh (1, num_devices) with ("batch", "model").
  - Shards Q/K/V column-parallel, O/down row-parallel across devices.

Known workarounds applied:
  - unflatten -> reshape monkey-patch: avoids dynamo graph break
  - prims::view_of -> clone: avoids XLA functionalization error
  - lifted tensor buffers: prevents c_lifted_tensor_* KeyError

Input:  video tokens [B, n_video, 128], audio tokens [B, n_audio, 128],
        video text [B, text_len, 3840], audio text [B, text_len, 3840],
        timestep, masks, spatial dims
Output: denoised video [B, n_video, 128], denoised audio [B, n_audio, 128]
"""

import os
import time

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from diffusers import LTX2VideoTransformer3DModel
from torch_xla.distributed.spmd import Mesh


def shard_transformer(model, mesh):
    """Apply Megatron-style TP sharding to the dual-stream DiT transformer.

    Column-parallel (Q/K/V, gate/up): weight sharded on dim 0 → ("model", None)
    Row-parallel (O, down): weight sharded on dim 1 → (None, "model")
    """
    shard_specs = {}

    for block in model.transformer_blocks:
        # Video self-attention: column-parallel Q/K/V, row-parallel O
        shard_specs[block.attn1.to_q.weight] = ("model", None)
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_v.weight] = ("model", None)
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")

        # Audio self-attention
        shard_specs[block.audio_attn1.to_q.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_k.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_v.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_out[0].weight] = (None, "model")

        # Video cross-attention (text -> video)
        shard_specs[block.attn2.to_q.weight] = ("model", None)
        shard_specs[block.attn2.to_k.weight] = ("model", None)
        shard_specs[block.attn2.to_v.weight] = ("model", None)
        shard_specs[block.attn2.to_out[0].weight] = (None, "model")

        # Audio cross-attention (text -> audio)
        shard_specs[block.audio_attn2.to_q.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_k.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_v.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_out[0].weight] = (None, "model")

        # Cross-modal: audio-to-video
        shard_specs[block.audio_to_video_attn.to_q.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_k.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_v.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_out[0].weight] = (None, "model")

        # Cross-modal: video-to-audio
        shard_specs[block.video_to_audio_attn.to_q.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_k.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_v.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_out[0].weight] = (None, "model")

        # Video FFN: GELU proj (up) column-parallel, linear out (down) row-parallel
        shard_specs[block.ff.net[0].proj.weight] = ("model", None)
        shard_specs[block.ff.net[2].weight] = (None, "model")

        # Audio FFN
        shard_specs[block.audio_ff.net[0].proj.weight] = ("model", None)
        shard_specs[block.audio_ff.net[2].weight] = (None, "model")

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)


def run_ltx2_transformer(tp=False):
    xr.set_device_type("TT")
    torch_xla.set_custom_compile_options({"optimization_level": 1})

    num_layers = 48 if tp else 4

    mesh = None
    if tp:
        os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
        xr.use_spmd()
        num_devices = xr.global_runtime_device_count()
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, (1, num_devices), ("batch", "model"))
        print(f"TP mode: {num_devices} devices, mesh (1, {num_devices})")

    device = torch_xla.device()

    # Apply all patches (attention, view_of, lifted tensors, conv3d)
    from ltx2_patches import apply_all_patches
    apply_all_patches()

    # Create minimal transformer with random weights
    model = LTX2VideoTransformer3DModel(
        num_layers=num_layers,
    ).to(torch.bfloat16)
    model.config.rope_type = "interleaved"  # avoid split rotary emb tracing bug
    model = model.eval()

    model = model.to(device)

    if tp and mesh is not None:
        shard_transformer(model, mesh)

    # Wrap model to clone outputs — avoids prims::view_of aliasing error
    # during functionalization. The transformer's outputs are views of internal
    # tensors, which triggers unsupported view_of prim.
    class CloneOutputWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, *args, **kwargs):
            out = self.inner(*args, **kwargs)
            return out.sample.clone(), out.audio_sample.clone()

    wrapper = CloneOutputWrapper(model)
    compiled = torch.compile(wrapper, backend="tt")

    # Small inputs for bringup
    num_frames, h, w = 2, 4, 4
    n_video = num_frames * h * w  # 32
    n_audio = 16
    text_len = 16

    hidden_states = torch.randn(1, n_video, 128, dtype=torch.bfloat16).to(device)
    audio_hidden_states = torch.randn(1, n_audio, 128, dtype=torch.bfloat16).to(device)
    encoder_hidden_states = torch.randn(1, text_len, 3840, dtype=torch.bfloat16).to(device)
    audio_encoder_hidden_states = torch.randn(1, text_len, 3840, dtype=torch.bfloat16).to(device)
    timestep = torch.tensor([500], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones(1, text_len, dtype=torch.long).to(device)
    audio_encoder_attention_mask = torch.ones(1, text_len, dtype=torch.long).to(device)

    kwargs = dict(
        hidden_states=hidden_states,
        audio_hidden_states=audio_hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        audio_encoder_hidden_states=audio_encoder_hidden_states,
        timestep=timestep,
        encoder_attention_mask=encoder_attention_mask,
        audio_encoder_attention_mask=audio_encoder_attention_mask,
        num_frames=num_frames,
        height=h,
        width=w,
        audio_num_frames=n_audio,
    )

    # Warm-up pass (compilation)
    mode_str = "TP" if tp else "replicated"
    print(f"Transformer ({num_layers}-layer DiT, {mode_str}): compiling...")
    with torch.no_grad():
        video_out, audio_out = compiled(**kwargs)
    torch_xla.sync(wait=True)
    print(f"  Video output shape: {video_out.shape}")
    print(f"  Audio output shape: {audio_out.shape}")

    # Timed pass
    print(f"Transformer ({num_layers}-layer DiT, {mode_str}): timed pass...")
    start = time.time()
    with torch.no_grad():
        video_out, audio_out = compiled(**kwargs)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Video output shape: {video_out.shape}")
    print(f"  Audio output shape: {audio_out.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return video_out, audio_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LTX-2 DiT Transformer bringup")
    parser.add_argument("--tp", action="store_true", help="Enable 4-way tensor parallel sharding")
    args = parser.parse_args()
    run_ltx2_transformer(tp=args.tp)
