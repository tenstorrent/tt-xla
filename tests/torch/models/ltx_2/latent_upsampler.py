# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

os.environ["TTXLA_LOGGER_LEVEL"] = "DEBUG"
os.environ["XLA_HLO_DEBUG"] = "1"
os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "DEBUG"


def run_latent_upsampler():
    """
    Test LTX2LatentUpsamplerModel in isolation.

    Uses random weights (no pretrained loading).
    Provides 2x spatial upscaling of video latents.

    Input: Video latent [B, 128, t, h, w]
    Output: Video latent [B, 128, t, 2*h, 2*w] (2x spatial upscaling)
    """
    xr.set_device_type("TT")
    device = torch_xla.device()

    upsampler = LTX2LatentUpsamplerModel().to(torch.bfloat16)
    upsampler = upsampler.eval().to(device)
    upsampler = torch.compile(upsampler, backend="tt")

    # Input: video latent [B, 128, t, h, w]
    latent = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16).to(device)

    with torch.no_grad():
        output = upsampler(latent)

    torch_xla.sync()
    print(f"Input shape: {list(latent.shape)}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: [1, 128, 2, 8, 8] (2x spatial upscaling)")


def load_model():
    upsampler = LTX2LatentUpsamplerModel().to(torch.bfloat16)
    return upsampler


def load_inputs():
    latent = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16)
    return latent


if __name__ == "__main__":
    print("Running LTX-2 Latent Upsampler test...")
    run_latent_upsampler()
