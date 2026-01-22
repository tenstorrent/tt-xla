# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runs CLIP Vision Encoder + IP-Adapter Resampler pipeline on TT hardware.
This is the image encoding portion of IP-Adapter Plus for SDXL.
"""

import torch
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

# Set up XLA runtime for TT backend
xr.set_device_type("TT")

# --- 2. LOAD BACKBONES ON CPU ---
print(f"Loading CLIP Vision Encoder and Processor in {dtype}...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    image_encoder_id, torch_dtype=dtype
)
processor = CLIPImageProcessor.from_pretrained(image_encoder_id)

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

# --- 4. PREPARE MODELS FOR TT HARDWARE ---
print("Compiling models for TT hardware...")
image_encoder.eval()
resampler.eval()

# Compile for TT backend
image_encoder.compile(backend="tt")
resampler.compile(backend="tt")

# Get TT device and move models
device = xm.xla_device()
image_encoder = image_encoder.to(device)
resampler = resampler.to(device)

# --- 5. RUN REAL IMAGE THROUGH THE FLOW ---
print("Processing input image...")
raw_image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
input_data = processor(images=raw_image, return_tensors="pt")

# Move input to device
pixel_values = input_data["pixel_values"].to(dtype).to(device)

with torch.no_grad():
    # Step A: Get CLIP hidden states
    clip_outputs = image_encoder(pixel_values=pixel_values, output_hidden_states=True)

    # Step B: Extract penultimate layer (Standard for IP-Adapter Plus)
    # Shape: [1, 257, 1280]
    patches = clip_outputs.hidden_states[-2]

    # Step C: Run through the Isolated Resampler
    # Shape: [1, 16, 2048]
    output_tokens = resampler(patches)


print("-" * 30)
print(f"Final IP-Adapter Tokens Shape: {output_tokens.shape}")
print(f"Dtype: {output_tokens.dtype}")
print(f"First 5 values of first token:\n{output_tokens[0, 0, :5]}")
