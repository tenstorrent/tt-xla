# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import AutoencoderKLLTX2Video

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_vae_encoder():
    """
    Test LTX-2 Video VAE encoder in isolation.

    Uses random weights (no pretrained loading) for fast, offline testing.

    Input: Video frames [B, 3, T, H, W]
    Output: Latent representation [B, 256, t, h, w]
            (128 channels for mean + 128 channels for logvar = 256)

    Compression: 8x temporal, 32x32 spatial
    Constraints:
    - Frame count T must be 8k+1 (9, 17, 25, 33, ...)
    - Height and width must be divisible by 32
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Create Video VAE with random weights
    vae = AutoencoderKLLTX2Video().to(torch.bfloat16)

    # Extract just the encoder
    model = vae.encoder
    model = model.eval().to(device)
    model = torch.compile(model, backend="tt")

    # Video input: [B, 3, T, H, W]
    # T=9 (8k+1), H=W=128 (divisible by 32)
    # Expected output: [1, 256, 2, 4, 4]
    #   latent_frames = (9-1)/8 + 1 = 2
    #   latent_h = 128/32 = 4, latent_w = 128/32 = 4
    #   channels = 128 (mean) + 128 (logvar) = 256
    sample_video = torch.randn(1, 3, 9, 128, 128, dtype=torch.bfloat16)
    sample_video = sample_video.to(device)

    with torch.no_grad():
        output = model(sample_video)

    torch_xla.sync()
    print(f"VAE Encoder output shape: {output.shape}")
    print(f"Expected shape: [1, 256, 2, 4, 4] (128 mean + 128 logvar)")


def load_model():
    vae = AutoencoderKLLTX2Video().to(torch.bfloat16)
    return vae.encoder


def load_inputs():
    sample_video = torch.randn(1, 3, 9, 128, 128, dtype=torch.bfloat16)
    return sample_video


if __name__ == "__main__":
    print("Running LTX-2 Video VAE Encoder test...")
    run_vae_encoder()
