# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LTX-2 Latent Upsampler — standalone bringup script.

Component: LTX2LatentUpsamplerModel
Memory: 0.93 GiB (bf16)
Sharding: Replicated (single device) — Conv3d-based spatial 2x upsampler
Hardware: Any single p150 chip (32 GiB DRAM)

Known blocker:
  - Conv3d requires symmetric padding (1,1,1) for 3x3x3 kernels but
    tt-metal only supports (0,x,x) temporal padding

Input:  [B, 128, t, h, w] video latent
Output: [B, 128, t, 2h, 2w] spatially upsampled latent (2x spatial, same temporal/channels)
"""

import time

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel

from conv3d_decompose import patch_conv3d_to_conv2d


def run_ltx2_latent_upsampler():
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Decompose Conv3d -> Conv2d to avoid tt-metal L1 overflow
    patch_conv3d_to_conv2d()

    # Load pretrained latent upsampler
    upsampler = LTX2LatentUpsamplerModel.from_pretrained(
        "Lightricks/LTX-2",
        subfolder="latent_upsampler",
        torch_dtype=torch.bfloat16,
    )
    upsampler = upsampler.eval()

    upsampler = upsampler.to(device)
    upsampler = torch.compile(upsampler, backend="tt")

    # Input: [1, 128, 2, 4, 4] -> Output: [1, 128, 2, 8, 8] (2x spatial upscale)
    latent_input = torch.randn(1, 128, 2, 4, 4, dtype=torch.bfloat16).to(device)

    # Warm-up pass (compilation)
    print("Latent Upsampler: warm-up pass (compilation)...")
    with torch.no_grad():
        output = upsampler(latent_input)
    torch_xla.sync(wait=True)
    print(f"  Output shape: {output.shape}")

    # Timed pass
    print("Latent Upsampler: timed pass...")
    start = time.time()
    with torch.no_grad():
        output = upsampler(latent_input)
    torch_xla.sync(wait=True)
    elapsed = time.time() - start
    print(f"  Output shape: {output.shape}")
    print(f"  Inference time: {elapsed:.3f}s")

    return output


if __name__ == "__main__":
    run_ltx2_latent_upsampler()
