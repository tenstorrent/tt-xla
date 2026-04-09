# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Audio VAE Decoder — standalone bringup script.

Component: AutoencoderKLLTX2Audio.decoder
Memory: ~0.05 GiB (bf16)
Sharding: Replicated (single device) — Conv2d-based, tiny model
Hardware: Any single p150 chip (32 GiB DRAM)

Input:  [B, 8, t_a, compressed_mel] audio latent (8 latent channels)
Output: [B, 2, ~t_a*4, mel_bins] reconstructed mel spectrogram (stereo)
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.models.autoencoders import AutoencoderKLLTX2Audio


def run_ltx2_audio_vae_decoder():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Load pretrained Audio VAE, extract decoder
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="audio_vae",
        torch_dtype=torch.bfloat16,
    )
    decoder = audio_vae.decoder.eval()
    del audio_vae

    decoder = decoder.to(device)
    decoder = torch.compile(decoder, backend="tt")

    # Input: audio latent [1, 8, 25, 16]
    # Output: mel spectrogram [1, 2, ~97, 64] (stereo, 4x temporal upsample, 4x mel upsample)
    latent_input = torch.randn(1, 8, 25, 16, dtype=torch.bfloat16).to(device)

    # Warm-up pass (compilation)
    print("Audio VAE Decoder: warm-up pass (compilation)...")
    with torch.no_grad():
        output = decoder(latent_input)
    torch_xla.sync(wait=True)
    print(f"  Output shape: {output.shape}")

    # Timed pass
    print("Audio VAE Decoder: timed pass...")
    start = time.time()
    with torch.no_grad():
        output = decoder(latent_input)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return output


if __name__ == "__main__":
    run_ltx2_audio_vae_decoder()
