# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import AutoencoderKLMochi

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_vae_encoder():
    """
    Test Mochi VAE encoder in isolation.

    Input: Video frames [B, 3, T, H, W]
    Output: Latent representation [B, 12, t, h, w]

    Mochi VAE specs:
    - Encodes RGB video to 12-channel latent space
    - 128x compression: 6x temporal, 8x8 spatial
    - Returns mean and logvar for VAE sampling

    Memory: Only loads VAE (~362M params) instead of full pipeline (~15B)
    """
    xr.set_device_type("TT")
    device = xm.xla_device()

    # Load ONLY the VAE (not the full pipeline!)
    # This loads ~362M params instead of ~15B params
    vae = AutoencoderKLMochi.from_pretrained(
        "genmo/mochi-1-preview", subfolder="vae", torch_dtype=torch.bfloat16
    )

    # Extract just the encoder
    model = vae.encoder
    model = model.eval().to(device)
    model = torch.compile(model, backend="tt")

    # Video input: [B, 3, T, H, W]
    # Using small test size: 13 frames at 256x256
    # Should produce latent: [1, 12, 3, 32, 32]
    sample_video = torch.randn(1, 3, 12, 128, 128, dtype=torch.bfloat16)
    sample_video = sample_video.to(device)

    with torch.no_grad():
        output = model(sample_video)

    print(f"VAE Encoder output shape: {output.shape}")
    print(f"Expected shape: [1, 24, 3, 32, 32] (12 for mean + 12 for logvar)")
    print(output)


if __name__ == "__main__":
    print("Running Mochi VAE Encoder test...")
    run_vae_encoder()
