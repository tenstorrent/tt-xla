# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Video VAE Decoder — standalone bringup script.

Component: AutoencoderKLLTX2Video.decoder
Memory: ~0.57 GiB (bf16), part of 1.14 GiB total VAE
Sharding: Replicated (single device) — Conv3d-based, not tensor-parallel friendly
Hardware: Any single p150 chip (32 GiB DRAM)

Known blockers:
  - Conv3d weight tiling causes ~24 GiB buffer allocation on n150 (exceeds 12 GiB)
  - c_out_block=32 incompatible with 48 output channels (48 % 32 != 0)
  - Conv3d temporal padding: tt-metal requires padding=0 but model needs padding=1
  On p150 (32 GiB), memory should be sufficient but Conv3d issues may persist.

Input:  [B, 128, t, h, w] video latent
Output: [B, 3, T, H, W] decoded video (T=(t-1)*8+1, H=h*32, W=w*32)
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import AutoencoderKLLTX2Video


def run_ltx2_video_decoder():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Load pretrained Video VAE, extract decoder
    vae = AutoencoderKLLTX2Video.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="vae",
        torch_dtype=torch.bfloat16,
    )
    decoder = vae.decoder.eval()
    del vae

    decoder = decoder.to(device)
    decoder = torch.compile(decoder, backend="tt")

    # Input: latent [1, 128, 2, 4, 4]
    # Output: video [1, 3, 9, 128, 128] (T=(2-1)*8+1=9, H=4*32=128, W=4*32=128)
    latent_input = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16).to(device)

    # Warm-up pass (compilation)
    print("Video VAE Decoder: warm-up pass (compilation)...")
    with torch.no_grad():
        output = decoder(latent_input)
    torch_xla.sync(wait=True)
    print(f"  Output shape: {output.shape}")

    # Timed pass
    print("Video VAE Decoder: timed pass...")
    start = time.time()
    with torch.no_grad():
        output = decoder(latent_input)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return output


if __name__ == "__main__":
    run_ltx2_video_decoder()
