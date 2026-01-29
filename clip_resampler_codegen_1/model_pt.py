# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
PyTorch golden reference for CLIP Vision Encoder + IP-Adapter Resampler.

This module provides:
- Model loading from HuggingFace (same architecture as TTNN codegen)
- Input generation (same input used during codegen)
- Golden output computation for PCC comparison
"""

import os

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image

# Model configuration (must match codegen source)
dtype = torch.bfloat16
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_adapter_weights_name = "ip-adapter-plus_sdxl_vit-h.bin"
MODEL_CACHE_PATH = "clip_resampler_sdxl.pt"

# Input image URL (same as codegen)
INPUT_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"


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


def get_model(cache_path=MODEL_CACHE_PATH):
    """Get model, loading from cache if available."""
    if cache_path and os.path.exists(cache_path):
        print(f"Loading model from cache: {cache_path}")
        model = torch.load(cache_path, weights_only=False)
        model.eval()
        return model

    print(f"Loading CLIP Vision Encoder from {image_encoder_id}...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_id, torch_dtype=dtype
    )

    print(f"Loading SDXL Pipeline from {model_id}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, image_encoder=image_encoder, torch_dtype=dtype
    )

    print("Loading IP-Adapter Plus weights...")
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models", weight_name=ip_adapter_weights_name
    )

    # Extract resampler
    resampler = pipe.unet.encoder_hid_proj.image_projection_layers[0]

    # Create combined module
    model = CLIPResamplerModule(image_encoder, resampler)
    model.eval()

    # Save to cache
    if cache_path:
        print(f"Saving model to cache: {cache_path}")
        torch.save(model, cache_path)

    return model


def get_input():
    """
    Get input pixel_values matching those used during codegen.
    Returns: torch.Tensor of shape [1, 3, 224, 224] in bfloat16
    """
    raw_image = load_image(INPUT_IMAGE_URL)
    processor = CLIPImageProcessor.from_pretrained(image_encoder_id)
    input_data = processor(images=raw_image, return_tensors="pt")
    pixel_values = input_data["pixel_values"].to(dtype)
    return pixel_values


def run_pytorch_inference(model=None, input_tensor=None):
    """
    Run PyTorch inference to get golden output.

    Args:
        model: Optional pre-loaded model
        input_tensor: Optional pre-loaded input tensor

    Returns:
        torch.Tensor: Output of shape [1, 16, 2048]
    """
    if model is None:
        model = get_model()
    if input_tensor is None:
        input_tensor = get_input()

    with torch.no_grad():
        output = model(input_tensor)

    return output
