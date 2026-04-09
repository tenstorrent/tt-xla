# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Audio VAE Encoder — standalone bringup script.

Component: AutoencoderKLLTX2Audio.encoder
Memory: ~0.05 GiB (bf16)
Sharding: Replicated (single device) — Conv2d-based, tiny model
Hardware: Any single p150 chip (32 GiB DRAM)

Input:  [B, 2, mel_frames, mel_bins] mel spectrogram (stereo, 2 channels)
Output: [B, 16, t_compressed, mel_compressed] (16 = 8 mean + 8 logvar)
        t_compressed = mel_frames/4, mel_compressed = mel_bins/4
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.models.autoencoders import AutoencoderKLLTX2Audio


def run_ltx2_audio_vae_encoder():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Load pretrained Audio VAE, extract encoder
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="audio_vae",
        torch_dtype=torch.bfloat16,
    )
    encoder = audio_vae.encoder.eval()
    del audio_vae

    encoder = encoder.to(device)
    encoder = torch.compile(encoder, backend="tt")

    # Input: stereo mel spectrogram [1, 2, 100, 64]
    # Output: [1, 16, 25, 16] (100/4=25, 64/4=16, 16 = 8 mean + 8 logvar)
    mel_input = torch.randn(1, 2, 100, 64, dtype=torch.bfloat16).to(device)

    # Warm-up pass (compilation)
    print("Audio VAE Encoder: warm-up pass (compilation)...")
    with torch.no_grad():
        output = encoder(mel_input)
    torch_xla.sync(wait=True)
    print(f"  Output shape: {output.shape}")

    # Timed pass
    print("Audio VAE Encoder: timed pass...")
    start = time.time()
    with torch.no_grad():
        output = encoder(mel_input)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return output


if __name__ == "__main__":
    run_ltx2_audio_vae_encoder()
