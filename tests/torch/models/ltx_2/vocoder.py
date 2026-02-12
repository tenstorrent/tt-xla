# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.pipelines.ltx2 import LTX2Vocoder

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_vocoder():
    """
    Test LTX2Vocoder (HiFi-GAN) in isolation.

    Uses random weights (no pretrained loading).
    Converts mel spectrograms from the Audio VAE decoder into audio waveforms.

    Architecture: Modified HiFi-GAN V1
    - Multi-scale upsampling with transposed convolutions
    - Multi-receptive field fusion (MRF)
    - Stereo output at 24kHz

    Input: 4D tensor [B, channels, mel_frames, mel_bins]
           where channels=2 (from Audio VAE decoder output)
    Output: Stereo waveform [B, 2, num_samples] at 24kHz
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    vocoder = LTX2Vocoder().to(torch.bfloat16)
    vocoder = vocoder.eval().to(device)
    vocoder = torch.compile(vocoder, backend="tt")

    # Vocoder expects 4D: [B, channels, time, mel_bins]
    # channels=2 (from Audio VAE decoder), time=100 frames, mel_bins=64
    mel = torch.randn(1, 2, 100, 64, dtype=torch.bfloat16).to(device)

    with torch.no_grad():
        output = vocoder(mel)

    torch_xla.sync()
    print(f"Vocoder output shape: {output.shape}")
    print(f"Expected: [1, 2, 24000] (stereo waveform at 24kHz)")


if __name__ == "__main__":
    print("Running LTX-2 Vocoder (HiFi-GAN) test...")
    run_vocoder()
