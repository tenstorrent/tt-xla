# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Vocoder (HiFi-GAN) — standalone bringup script.

Component: LTX2Vocoder
Memory: 0.10 GiB (bf16)
Sharding: Replicated (single device) — small transposed-conv model
Hardware: Any single p150 chip (32 GiB DRAM)

Converts mel spectrograms to stereo audio waveforms at 24kHz.

Input:  [B, 2, time, 64] mel spectrogram (stereo)
Output: [B, 2, samples] stereo waveform (~time * hop_length samples)
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.pipelines.ltx2 import LTX2Vocoder


def run_ltx2_vocoder():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Load pretrained vocoder
    vocoder = LTX2Vocoder.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="vocoder",
        torch_dtype=torch.bfloat16,
    )
    vocoder = vocoder.eval()

    vocoder = vocoder.to(device)
    vocoder = torch.compile(vocoder, backend="tt")

    # Input: stereo mel spectrogram [1, 2, 100, 64]
    # Output: stereo waveform [1, 2, ~24000] at 24kHz
    mel_input = torch.randn(1, 2, 100, 64, dtype=torch.bfloat16).to(device)

    # Warm-up pass (compilation)
    print("Vocoder (HiFi-GAN): warm-up pass (compilation)...")
    with torch.no_grad():
        output = vocoder(mel_input)
    torch_xla.sync(wait=True)
    print(f"  Output shape: {output.shape}")

    # Timed pass
    print("Vocoder (HiFi-GAN): timed pass...")
    start = time.time()
    with torch.no_grad():
        output = vocoder(mel_input)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return output


if __name__ == "__main__":
    run_ltx2_vocoder()
