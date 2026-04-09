# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Dual-Stream DiT Transformer — standalone bringup script with 4-chip tensor parallelism.

Component: LTX2VideoTransformer3DModel
Memory: 35.17 GiB (bf16) — must be sharded across multiple chips
Sharding: 4-way tensor parallel across bhqb (4 x p150)
  - Per-device: 35.17/4 = 8.79 GiB weights, leaving ~23.21 GiB for activations
  - Video stream: 32 heads / 4 = 8 heads/device, inner_dim=4096
  - Audio stream: 32 heads / 4 = 8 heads/device, inner_dim=2048
  - FFN: video 16384 intermediate, audio 8192 intermediate — both divisible by 4

Architecture: Asymmetric dual-stream DiT (48 layers)
  - Video: 32 heads x 128 head_dim = 4096 inner_dim, cross_attn_dim=4096
  - Audio: 32 heads x 64 head_dim = 2048 inner_dim, cross_attn_dim=2048
  - Cross-modal: bidirectional audio<->video attention
  - 3D RoPE for video, 1D RoPE for audio

Known blocker:
  - `KeyError: c_lifted_tensor_1` — torch_xla graph partitioner doesn't preserve
    shared get_attr nodes across partition boundaries (critical, blocks core denoising)

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
from torch_xla.distributed.spmd import Mesh
from diffusers import LTX2VideoTransformer3DModel


def shard_transformer(model, mesh):
    """Apply tensor-parallel sharding to the dual-stream transformer."""
    shard_specs = {}

    for block in model.transformer_blocks:
        # === Video self-attention (attn1): 32 heads, dim 4096 ===
        shard_specs[block.attn1.to_q.weight] = ("model", None)  # column-parallel
        shard_specs[block.attn1.to_k.weight] = ("model", None)
        shard_specs[block.attn1.to_v.weight] = ("model", None)
        shard_specs[block.attn1.to_out[0].weight] = (None, "model")  # row-parallel

        # === Audio self-attention (audio_attn1): 32 heads, dim 2048 ===
        shard_specs[block.audio_attn1.to_q.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_k.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_v.weight] = ("model", None)
        shard_specs[block.audio_attn1.to_out[0].weight] = (None, "model")

        # === Video cross-attention (attn2): text -> video ===
        shard_specs[block.attn2.to_q.weight] = ("model", None)
        shard_specs[block.attn2.to_k.weight] = ("model", None)
        shard_specs[block.attn2.to_v.weight] = ("model", None)
        shard_specs[block.attn2.to_out[0].weight] = (None, "model")

        # === Audio cross-attention (audio_attn2): text -> audio ===
        shard_specs[block.audio_attn2.to_q.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_k.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_v.weight] = ("model", None)
        shard_specs[block.audio_attn2.to_out[0].weight] = (None, "model")

        # === Cross-modal: audio-to-video attention ===
        # Q comes from video (4096), K/V from audio (2048), output goes to video (4096)
        shard_specs[block.audio_to_video_attn.to_q.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_k.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_v.weight] = ("model", None)
        shard_specs[block.audio_to_video_attn.to_out[0].weight] = (None, "model")

        # === Cross-modal: video-to-audio attention ===
        # Q from audio (2048), K/V from video (4096), output goes to audio (2048)
        shard_specs[block.video_to_audio_attn.to_q.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_k.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_v.weight] = ("model", None)
        shard_specs[block.video_to_audio_attn.to_out[0].weight] = (None, "model")

        # === Video FFN: 4096 -> 16384 -> 4096 (GEGLU, so gate proj outputs 16384) ===
        shard_specs[block.ff.net[0].proj.weight] = ("model", None)  # column-parallel (GEGLU gate+up fused)
        shard_specs[block.ff.net[2].weight] = (None, "model")  # row-parallel (down proj)

        # === Audio FFN: 2048 -> 8192 -> 2048 ===
        shard_specs[block.audio_ff.net[0].proj.weight] = ("model", None)
        shard_specs[block.audio_ff.net[2].weight] = (None, "model")

    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)


def run_ltx2_transformer():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()
    assert num_devices >= 4, f"Transformer requires 4 devices, found {num_devices}"

    # Create mesh for tensor parallelism
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    device = torch_xla.device()

    # Load pretrained transformer
    print("Loading LTX-2 transformer from Lightricks/LTX-2...")
    model = LTX2VideoTransformer3DModel.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    # Override rope_type to avoid split rotary emb tracing bug
    model.config.rope_type = "interleaved"
    model = model.eval()

    # Move to device and apply sharding
    model = model.to(device)
    shard_transformer(model, mesh)

    # Compile
    compiled_model = torch.compile(model, backend="tt")

    # Prepare inputs for a small video generation:
    # 512x320 resolution, 49 frames -> latent: t=7, h=16, w=10
    # n_video_tokens = t * h * w = 7 * 16 * 10 = 1120
    num_frames = 7
    height = 16
    width = 10
    n_video = num_frames * height * width  # 1120

    # Audio tokens: depends on audio duration, use representative count
    n_audio = 64
    text_len = 128
    caption_channels = 3840

    hidden_states = torch.randn(1, n_video, 128, dtype=torch.bfloat16).to(device)
    audio_hidden_states = torch.randn(1, n_audio, 128, dtype=torch.bfloat16).to(device)
    encoder_hidden_states = torch.randn(1, text_len, caption_channels, dtype=torch.bfloat16).to(device)
    audio_encoder_hidden_states = torch.randn(1, text_len, caption_channels, dtype=torch.bfloat16).to(device)
    timestep = torch.tensor([500], dtype=torch.long).to(device)
    encoder_attention_mask = torch.ones(1, text_len, dtype=torch.long).to(device)
    audio_encoder_attention_mask = torch.ones(1, text_len, dtype=torch.long).to(device)

    # Warm-up pass (compilation)
    print("Transformer (48-layer DiT, 4-chip TP): warm-up pass (compilation)...")
    with torch.no_grad():
        output = compiled_model(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
        )
    torch_xla.sync(wait=True)
    print(f"  Video output shape: {output.sample.shape}")
    print(f"  Audio output shape: {output.audio_sample.shape}")

    # Timed pass
    print("Transformer (48-layer DiT, 4-chip TP): timed pass...")
    start = time.time()
    with torch.no_grad():
        output = compiled_model(
            hidden_states=hidden_states,
            audio_hidden_states=audio_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            audio_encoder_hidden_states=audio_encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            audio_encoder_attention_mask=audio_encoder_attention_mask,
            num_frames=num_frames,
            height=height,
            width=width,
        )
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Video output shape: {output.sample.shape}")
    print(f"  Audio output shape: {output.audio_sample.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return output


if __name__ == "__main__":
    xr.set_device_type("TT")
    run_ltx2_transformer()
