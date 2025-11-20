# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import AutoencoderKLMochi

os.environ["LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"


def run_vae_decoder():
    """
    Test Mochi VAE decoder in isolation.

    Input: Latent representation [B, 12, t, h, w]
    Output: Video frames [B, 3, T, H, W]

    Mochi VAE specs:
    - 12 latent channels
    - 128x compression: 6x temporal, 8x8 spatial
    - Example: 24 frames @ 480x848 -> latents [1, 12, 4, 60, 106]
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Load ONLY the VAE (not the full pipeline!)
    # This loads ~362M params instead of ~15B params
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16
    )

    enable_tiling = False

    if enable_tiling:
        vae.enable_tiling(
            tile_sample_min_height=120,  # Split 480 into 4 tiles
            tile_sample_min_width=212,  # Split 848 into 4 tiles
            tile_sample_stride_height=100,  # 20 pixel overlap
            tile_sample_stride_width=180,  # 32 pixel overlap
        )

    # VAE decoder: [B, 12, t, h, w] → [B, 3, T, H, W]
    # For 24 frames @ 480x848 which is 1s of video at 24fps
    # Ecoder has 6x temporal compression and 8x8 spatial compression
    # So for the video [1, 3, 24, 480, 848] we get
    # Latent: [1, 12, 4, 60, 106] (24/6≈4, 480/8=60, 848/8=106)
    latent = torch.randn(1, 12, 4, 60, 106, dtype=torch.bfloat16)
    latent = latent.to(device)

    # Extract just the decoder
    model = vae.decoder
    model = model.eval().to(device)
    model = torch.compile(model, backend="tt")

    with torch.no_grad():
        # Normalize latents (VAE expects normalized input)
        # Mochi normalizes with channel-wise std
        vae_std = torch.tensor(
            [
                0.925,
                0.934,
                0.946,
                0.939,
                0.961,
                1.033,
                0.979,
                1.024,
                0.983,
                1.046,
                0.964,
                1.004,
            ],
            dtype=torch.bfloat16,
            device=device,
        )
        latent_normalized = latent / vae_std.view(1, 12, 1, 1, 1)

        # run just decoder forward pass
        output = model(latent_normalized)

        # run full pipeline forward pass including tiling as memory optimization
        # output = vae.decode(latent_normalized)

    print(f"Expected shape: [1, 3, 24, 480, 848]")
    torch_xla.sync()
    print(f"Got output shape: {output[0].shape}")


if __name__ == "__main__":
    print("Running Mochi VAE Decoder test...")
    run_vae_decoder()
