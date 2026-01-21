# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image

# --- 1. CONFIGURATION ---
device = "cpu"
dtype = torch.bfloat16
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_adapter_weights_name = "ip-adapter-plus_sdxl_vit-h.bin"

# --- 2. LOAD BACKBONES ---
print(f"Loading CLIP Vision Encoder and Processor in {dtype}...")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    image_encoder_id, torch_dtype=dtype
).to(device)
processor = CLIPImageProcessor.from_pretrained(image_encoder_id)

print("Loading SDXL Pipeline (extracting Resampler architecture)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id, image_encoder=image_encoder, torch_dtype=dtype, device_map=device
)

# --- 3. ATTACH IP-ADAPTER & EXTRACT RESAMPLER ---
print("Loading IP-Adapter Plus weights...")
pipe.load_ip_adapter(
    "h94/IP-Adapter", subfolder="sdxl_models", weight_name=ip_adapter_weights_name
)

# Isolate the exact Resampler module used by the library
resampler = pipe.unet.encoder_hid_proj.image_projection_layers[0]
resampler.eval()

# --- 4. RUN REAL IMAGE THROUGH THE FLOW ---
print("Processing input image...")
raw_image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
input_data = processor(images=raw_image, return_tensors="pt").to(device, dtype=dtype)

with torch.no_grad():
    # Step A: Get CLIP hidden states
    # Note: clip_model(**input_data) unpacks 'pixel_values' automatically
    clip_outputs = image_encoder(**input_data, output_hidden_states=True)

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

# --- 5. OPTIONAL: SAVE FOR CONVERTER ---
# torch.save(resampler.state_dict(), "isolated_resampler_bf16.pt")
