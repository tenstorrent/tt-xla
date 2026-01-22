# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runs CLIP Vision Encoder + IP-Adapter Resampler pipeline on TT hardware.
This is the image encoding portion of IP-Adapter Plus for SDXL.
"""

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image

# --- 1. CONFIGURATION ---
dtype = torch.bfloat16
torch_xla.set_custom_compile_options({"optimization_level": 0})
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_adapter_weights_name = "ip-adapter-plus_sdxl_vit-h.bin"


class CLIPResamplerModule(nn.Module):
    """Combined CLIP Vision Encoder + IP-Adapter Resampler module."""

    def __init__(self, image_encoder, resampler):
        super().__init__()
        self.image_encoder = image_encoder
        self.resampler = resampler

    def forward(self, pixel_values):
        # Get CLIP hidden states
        clip_outputs = self.image_encoder(
            pixel_values=pixel_values, output_hidden_states=True
        )
        # Extract penultimate layer (Standard for IP-Adapter Plus)
        # Shape: [batch, 257, 1280]
        patches = clip_outputs.hidden_states[-2]
        # Run through the Resampler
        # Shape: [batch, 16, 2048]
        output_tokens = self.resampler(patches)
        return output_tokens


def get_model():

    # --- 2. LOAD BACKBONES ON CPU ---
    print(f"Loading CLIP Vision Encoder and Processor in {dtype}...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_id, torch_dtype=dtype
    )

    print("Loading SDXL Pipeline (extracting Resampler architecture)...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, image_encoder=image_encoder, torch_dtype=dtype
    )

    # --- 3. ATTACH IP-ADAPTER & EXTRACT RESAMPLER ---
    print("Loading IP-Adapter Plus weights...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models", weight_name=ip_adapter_weights_name
    )

    # Isolate the exact Resampler module used by the library
    resampler = pipe.unet.encoder_hid_proj.image_projection_layers[0]

    # --- 4. CREATE COMBINED MODULE & PREPARE FOR TT HARDWARE ---
    print("Creating combined CLIP + Resampler module...")
    clip_resampler = CLIPResamplerModule(image_encoder, resampler)
    clip_resampler.eval()

    return clip_resampler


def get_input():
    raw_image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    processor = CLIPImageProcessor.from_pretrained(image_encoder_id)
    input_data = processor(images=raw_image, return_tensors="pt")
    pixel_values = input_data["pixel_values"].to(dtype)
    return input_data


def run_on_cpu():
    clip_resampler = get_model()
    input = get_input()

    device = xm.xla_device()

    input = input.to(device)
    clip_resampler = clip_resampler.to(device)

    print("Running on CPU...")
    output = clip_resampler(**input)
    print("Finished running on CPU")

    return output


def run_on_tt():
    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    clip_resampler = get_model()
    clip_resampler.compile(backend="tt")

    # Get TT device and move model
    device = xm.xla_device()
    clip_resampler = clip_resampler.to(device)

    # Move input to device
    input = get_input()
    input = input.to(device)

    print("Running on TT...")
    output = clip_resampler(input["pixel_values"])
    print("Finished running on TT")
    return output


def main():
    output_pt = run_on_cpu()
    output_tt = run_on_tt()

    # Print shapes
    print(f"Output pt shape: {output_pt.shape}")
    print(f"Output tt shape: {output_tt.shape}")

    # Print dtypes
    print(f"Output pt dtype: {output_pt.dtype}")
    print(f"Output tt dtype: {output_tt.dtype}")

    # Print first 5 values of first token
    print(f"Output pt first 5 values of first token: {output_pt[0, 0, :5]}")
    print(f"Output tt first 5 values of first token: {output_tt[0, 0, :5]}")

    # Compare outputs
    assert torch.allclose(output_pt, output_tt), "Outputs are not close"
    print("Outputs are close")


if __name__ == "__main__":
    main()
