# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Video VAE Encoder — standalone bringup script.

Component: AutoencoderKLLTX2Video.encoder
Memory: ~0.57 GiB (bf16), part of 1.14 GiB total VAE
Sharding: Replicated (single device) — Conv3d-based, not tensor-parallel friendly
Hardware: Any single p150 chip (32 GiB DRAM)

Input:  [B, 3, T, H, W] video frames (T must be 8k+1, H/W divisible by 32)
Output: [B, 256, t, h, w] latent (256 = 128 mean + 128 logvar, t=(T-1)/8+1, h=H/32, w=W/32)
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import AutoencoderKLLTX2Video


def run_ltx2_video_encoder():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Load pretrained Video VAE, extract encoder
    vae = AutoencoderKLLTX2Video.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    encoder = vae.encoder.eval()
    del vae

    encoder = encoder.to(device)
    encoder = torch.compile(encoder, backend="tt")

    # Input: 9 frames at 128x128 (minimal valid size: T=8k+1=9, H/W divisible by 32)
    # Output: [1, 256, 2, 4, 4] (t=(9-1)/8+1=2, h=128/32=4, w=128/32=4)
    video_input = torch.randn(1, 3, 9, 128, 128, dtype=torch.bfloat16).to(device)

    # Warm-up pass (compilation)
    print("Video VAE Encoder: warm-up pass (compilation)...")
    with torch.no_grad():
        output = encoder(video_input)
    torch_xla.sync(wait=True)
    print(f"  Output shape: {output.shape}")

    # Timed pass
    print("Video VAE Encoder: timed pass...")
    start = time.time()
    with torch.no_grad():
        output = encoder(video_input)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return output


if __name__ == "__main__":
    run_ltx2_video_encoder()
