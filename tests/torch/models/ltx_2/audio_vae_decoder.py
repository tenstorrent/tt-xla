# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.models.autoencoders import AutoencoderKLLTX2Audio

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_audio_vae_decoder():
    """
    Test LTX-2 Audio VAE decoder in isolation.

    Uses random weights (no pretrained loading).
    Decodes audio latent to mel spectrogram.

    Audio VAE specs:
    - latent_channels: 8 (NOT 128)
    - in_channels: 2, output_channels: 2
    - Temporal compression: 4x
    - Mel compression: 4x (64 mel bins -> 16 compressed)

    Input: Latent [B, 8, t_a, 16]
    Output: Mel spectrogram [B, 2, ~t_a*4, 64]
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    audio_vae = AutoencoderKLLTX2Audio().to(torch.bfloat16)

    model = audio_vae.decoder
    model = model.eval().to(device)
    model = torch.compile(model, backend="tt")

    # Audio latent: [B, 8, t_a, 16]
    # 8 latent channels, 25 temporal frames, 16 compressed mel bins
    latent = torch.randn(1, 8, 25, 16, dtype=torch.bfloat16).to(device)

    with torch.no_grad():
        output = model(latent)

    torch_xla.sync()
    print(f"Audio VAE Decoder output shape: {output.shape}")
    print(f"Expected: [1, 2, 97, 64]")


def load_model():
    audio_vae = AutoencoderKLLTX2Audio().to(torch.bfloat16)
    return audio_vae


def load_inputs():
    latent = torch.randn(1, 8, 25, 16, dtype=torch.bfloat16)
    return latent


if __name__ == "__main__":
    print("Running LTX-2 Audio VAE Decoder test...")
    run_audio_vae_decoder()
