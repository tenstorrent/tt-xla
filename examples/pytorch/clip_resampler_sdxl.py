# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Runs CLIP Vision Encoder + IP-Adapter Resampler pipeline on TT hardware.
This is the image encoding portion of IP-Adapter Plus for SDXL.
"""

import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image
from tt_torch import codegen_py

# CONFIG
compile_options = {
    "optimization_level": 1,
    "codegen_try_recover_structure": True,
}
EXPORT_PATH = "clip_resampler_codegen"
torch_xla.set_custom_compile_options(compile_options)

dtype = torch.bfloat16
model_id = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_adapter_weights_name = "ip-adapter-plus_sdxl_vit-h.bin"
MODEL_CACHE_PATH = "clip_resampler_sdxl.pt"


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


def save_model(model, path=MODEL_CACHE_PATH):
    """Save the entire CLIPResamplerModule to disk."""
    print(f"Saving model to {path}...")
    torch.save(model, path)
    print(f"Model saved to {path}")


def load_model(path=MODEL_CACHE_PATH):
    """Load CLIPResamplerModule directly from disk."""
    print(f"Loading model from {path}...")
    clip_resampler = torch.load(path, weights_only=False)
    clip_resampler.eval()
    print(f"Model loaded from {path}")
    return clip_resampler


def get_model(cache_path=MODEL_CACHE_PATH):
    """Get model, loading from cache if available."""
    if cache_path and os.path.exists(cache_path):
        return load_model(cache_path)

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

    # Save to cache for future use
    if cache_path:
        save_model(clip_resampler, cache_path)

    return clip_resampler


def get_input():
    raw_image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    processor = CLIPImageProcessor.from_pretrained(image_encoder_id)
    input_data = processor(images=raw_image, return_tensors="pt")
    input_data["pixel_values"] = input_data["pixel_values"].to(dtype)
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
    input_data = get_input()
    input_tensor = input_data["pixel_values"]
    input_tensor = input_tensor.to(device)

    print("Running on TT...")
    output = clip_resampler(input_tensor)
    print("Finished running on TT")
    return output


def run_codegen():
    """Generate Python code for the CLIPResamplerModule."""

    # Set up XLA runtime for TT backend
    xr.set_device_type("TT")

    clip_resampler = get_model()

    input_data = get_input()
    input_tensor = input_data["pixel_values"]

    print(f"Generating code to {EXPORT_PATH}...")
    codegen_py(
        clip_resampler,
        input_tensor,
        export_path=EXPORT_PATH,
        compiler_options=compile_options,
    )
    print(f"Code generated to {EXPORT_PATH}")


def get_shape_str(tensor_or_tuple):
    """Get shape string from a tensor or tuple of tensors."""
    if tensor_or_tuple is None:
        return "None"
    if isinstance(tensor_or_tuple, torch.Tensor):
        return str(list(tensor_or_tuple.shape))
    if isinstance(tensor_or_tuple, (tuple, list)):
        shapes = []
        for t in tensor_or_tuple:
            if isinstance(t, torch.Tensor):
                shapes.append(str(list(t.shape)))
            elif t is None:
                shapes.append("None")
            else:
                shapes.append(type(t).__name__)
        return f"({', '.join(shapes)})"
    return type(tensor_or_tuple).__name__


def print_module_structure():
    """Print the hierarchical structure of the model's modules with output shapes."""
    clip_resampler = get_model()
    input_data = get_input()

    # Collect shapes via hooks (keyed by module name)
    shapes = {}

    def make_hook(name):
        def hook(module, input, output):
            shapes[name] = get_shape_str(output)

        return hook

    # Register hooks on all modules
    handles = []
    for name, module in clip_resampler.named_modules():
        handle = module.register_forward_hook(make_hook(name))
        handles.append(handle)

    # Run forward pass to capture shapes
    with torch.no_grad():
        clip_resampler(input_data["pixel_values"])

    # Remove hooks
    for handle in handles:
        handle.remove()

    # Print in hierarchical order (named_modules order)
    print("\n" + "=" * 100)
    print("MODEL STRUCTURE (with output shapes)")
    print("=" * 100)
    for name, module in clip_resampler.named_modules():
        depth = name.count(".") if name else 0
        indent = "  " * depth
        module_name = name.split(".")[-1] if name else "(root)"
        shape = shapes.get(name, "?")
        print(f"{indent}{module_name}: {module.__class__.__name__} -> {shape}")
    print("=" * 100 + "\n")


def main():
    print_module_structure()
    run_codegen()
    return

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
