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


def run_audio_vae_encoder():
    """
    Test LTX-2 Audio VAE encoder in isolation.

    Uses random weights (no pretrained loading).
    Encodes mel spectrograms to audio latent space.

    Audio VAE specs:
    - latent_channels: 8 (NOT 128)
    - in_channels: 2, output_channels: 2
    - Temporal compression: 4x
    - Mel compression: 4x

    Input: Mel spectrogram [B, 2, mel_frames, 64]
    Output: Latent [B, 16, mel_frames/4, 16]  (8 mean + 8 logvar = 16)
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    audio_vae = AutoencoderKLLTX2Audio().to(torch.bfloat16)

    model = audio_vae.encoder
    model = model.eval().to(device)
    model = torch.compile(model, backend="tt")

    # Mel spectrogram: [B, 2, mel_frames, mel_bins]
    # 2 channels, 100 mel frames (~1 second of audio), 64 mel bins
    mel_input = torch.randn(1, 2, 100, 64, dtype=torch.bfloat16).to(device)

    with torch.no_grad():
        output = model(mel_input)

    torch_xla.sync()
    print(f"Audio VAE Encoder output shape: {output.shape}")
    print(f"Expected: [1, 16, 25, 16] (8 mean + 8 logvar, 100/4=25, 64/4=16)")


if __name__ == "__main__":
    print("Running LTX-2 Audio VAE Encoder test...")
    run_audio_vae_encoder()
