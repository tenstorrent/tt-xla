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


def calculate_output_shape(latent_shape):
    """
    LTX-2 Video VAE compression:
    - Temporal: 8x  (latent_frames -> (latent_frames-1)*8 + 1 output frames)
    - Spatial: 32x  (latent_h * 32, latent_w * 32)
    """
    return (
        latent_shape[0],
        3,
        (latent_shape[2] - 1) * 8 + 1,
        latent_shape[3] * 32,
        latent_shape[4] * 32,
    )


def run_vae_decoder():
    """
    Test LTX-2 Video VAE decoder in isolation.

    Uses random weights (no pretrained loading) for fast, offline testing.
    Architecture matches the pretrained model exactly via default constructor.

    Input: Latent representation [B, 128, t, h, w]
    Output: Video frames [B, 3, T, H, W]

    Compression: 8x temporal, 32x32 spatial
    Example: latent [1, 128, 2, 4, 4] -> video [1, 3, 9, 128, 128]
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Create Video VAE with random weights
    vae = AutoencoderKLLTX2Video().to(torch.bfloat16)

    # VAE decoder: [B, 128, t, h, w] -> [B, 3, T, H, W]
    latent = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16)
    latent = latent.to(device)

    model = vae.decoder
    model = model.eval().to(device)
    model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = model(latent)

    expected = calculate_output_shape(latent.shape)
    print(f"Expected shape: {expected}")
    torch_xla.sync()
    print(f"Got output shape: {output.shape}")


def load_model():
    vae = AutoencoderKLLTX2Video().to(torch.bfloat16)
    return vae.decoder


def load_inputs():
    latent = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16)
    return latent


if __name__ == "__main__":
    print("Running LTX-2 Video VAE Decoder test...")
    run_vae_decoder()
