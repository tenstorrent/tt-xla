# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import AutoencoderKL, StableDiffusionPipeline
from diffusers.image_processor import VaeImageProcessor


@pytest.mark.parametrize("sample_size", [16, 32])
def test_vae(sample_size):
    """Test VAE decoder of stable diffusion 1.4"""
    """image = self.vae.decode(latents / self.vae.config.scaling_factor,
            return_dict=False, generator=generator)[0]"""
    torch.manual_seed(42)

    print(f"Testing VAE decode performance for sample_size={sample_size}")

    # Load pipeline to get VAE
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Create dummy latents (what Unet would output)
    batch_size = 1
    num_channels_latents = pipe.unet.config.in_channels  # 4
    latents = torch.randn(
        batch_size, num_channels_latents, sample_size, sample_size, dtype=torch.bfloat16
    )

    # Scale latents (as done in pipeline)
    latents = latents / pipe.vae.config.scaling_factor

    print(f"Input latents shape: {latents.shape}")
    print(f"Expected output shape: {sample_size * 8}x{sample_size * 8}")

    print("Compiling VAE for TT device...")
    xr.set_device_type("TT")
    device = torch_xla.device()

    latents = latents.to(device)
    pipe.vae = pipe.vae.eval()
    pipe.vae.to(device)
    pipe.vae = torch.compile(pipe.vae, backend="tt")
    print("VAE compiled for TT device, device type: ", pipe.vae.device)

    with torch.no_grad():
        image = pipe.vae.decode(latents, return_dict=False)[0]

    image = image.to("cpu")

    print(f"Output image shape: {image.shape}")

    # Save the decoded image
    image.save(f"vae_decode_test_{sample_size}.png")
