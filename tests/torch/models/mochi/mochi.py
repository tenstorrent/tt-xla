# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import MochiPipeline

# Enable HLO debug output
os.environ["XLA_HLO_DEBUG"] = "1"


def run_full_pipeline():
    """
    Test full Mochi pipeline (optional - runs all components together).

    This is the complete text-to-video generation pipeline:
    1. Encode text with T5-XXL
    2. Generate latents with DiT (64 sampling steps)
    3. Decode latents with VAE decoder
    """
    xr.set_device_type("TT")

    # Load full pipeline
    pipeline = MochiPipeline.from_pretrained(
        "genmo/mochi-1-preview", torch_dtype=torch.float16, variant="bf16"
    )

    device = xm.xla_device()
    pipeline = pipeline.to(device)

    # Generate video from text prompt
    prompt = "A cat playing piano"

    with torch.no_grad():
        video = pipeline(
            prompt=prompt,
            num_frames=13,  # Small test size (must be 6k+1: 7, 13, 19, 25, 31, 37...)
            height=256,
            width=256,
            num_inference_steps=10,  # Reduced for testing (default: 64)
            guidance_scale=6.0,
        ).frames[0]

    print(f"Generated video shape: {video.shape}")
    print(f"Expected shape: [13, 256, 256, 3]")


if __name__ == "__main__":
    print("Running Mochi Full Pipeline test...")
    run_full_pipeline()
