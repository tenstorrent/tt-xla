# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
import time

import torch
import torch.nn as nn
import ttnn
import utils
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image

from model import CLIPResampler

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_id = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
ip_adapter_weights_name = "ip-adapter-plus_sdxl_vit-h.bin"
MODEL_CACHE_PATH = "clip_resampler_sdxl.pt"
WEIGHT_MAP_PATH = "weight_map.json"
dtype = torch.bfloat16


def load_inputs_from_pytorch(state_dict):
    """
    Load model weights from PyTorch state_dict instead of tensorbin files.

    Args:
        state_dict: PyTorch model state_dict

    Returns:
        List of TTNN tensors. The input tensor slot (index 390) contains None
        and should be passed separately to forward() as pixel_values.
    """
    # Load tensor configuration
    with open("tensor_load_config.json", "r") as f:
        tensor_config = json.load(f)

    device = utils.DeviceGetter.get_device((1, 1))
    dram_memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )

    # Sort by tensor_idx to ensure correct order
    sorted_configs = sorted(tensor_config.items(), key=lambda x: x[1]["tensor_idx"])

    result = []
    for arg_name, config in sorted_configs:
        weight_name = config["weight_name"]
        layout = getattr(ttnn.Layout, config["layout"])
        ttnn_dtype = getattr(ttnn.DataType, config["dtype"])
        on_device = config["on_device"]

        if weight_name == "__INPUT__":
            # Input tensor is passed separately to forward() as pixel_values
            result.append(None)

        elif weight_name == "__POSITION_IDS__":
            # Generate position IDs [0, 1, 2, ..., 256]
            pos_ids = torch.arange(257, dtype=torch.int32).unsqueeze(0)
            ttnn_tensor = ttnn.from_torch(pos_ids, dtype=ttnn_dtype, layout=layout)
            if on_device:
                ttnn_tensor = ttnn.to_device(ttnn_tensor, device, dram_memory_config)
            result.append(ttnn_tensor)

        elif weight_name is None:
            raise ValueError(f"Unknown tensor {arg_name} has no weight mapping")

        else:
            # Regular weight from state_dict
            ttnn_tensor = utils.load_weight_from_pytorch(
                state_dict,
                weight_name,
                layout,
                ttnn_dtype,
                device if on_device else None,
                dram_memory_config if on_device else None,
            )
            result.append(ttnn_tensor)

    return result


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


def get_input():
    raw_image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    processor = CLIPImageProcessor.from_pretrained(image_encoder_id)
    input_data = processor(images=raw_image, return_tensors="pt")
    input_data["pixel_values"] = input_data["pixel_values"].to(dtype)
    return input_data


def save_pt_model(model, path=MODEL_CACHE_PATH):
    """Save the entire CLIPResamplerModule to disk."""
    print(f"Saving model to {path}...")
    torch.save(model, path)
    print(f"Model saved to {path}")


def load_pt_model(path=MODEL_CACHE_PATH):
    """Load CLIPResamplerModule directly from disk."""
    print(f"Loading model from {path}...")
    clip_resampler = torch.load(path, weights_only=False)
    clip_resampler.eval()
    print(f"Model loaded from {path}")
    return clip_resampler


def get_pt_model(cache_path=MODEL_CACHE_PATH):
    """Get model, loading from cache if available."""
    if cache_path and os.path.exists(cache_path):
        return load_pt_model(cache_path)

    # Load CLIP Vision Encoder and Processor
    print(f"Loading CLIP Vision Encoder and Processor in {dtype}...")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        image_encoder_id, torch_dtype=dtype
    )

    # Load SDXL Pipeline (extracting Resampler architecture)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id, image_encoder=image_encoder, torch_dtype=dtype
    )

    # Attach IP-Adapter & Extract Resampler
    pipe.load_ip_adapter(
        "h94/IP-Adapter", subfolder="sdxl_models", weight_name=ip_adapter_weights_name
    )

    # Isolate the exact Resampler module used by the SDXL pipeline
    resampler = pipe.unet.encoder_hid_proj.image_projection_layers[0]

    # Create combined CLIP + Resampler module
    clip_resampler = CLIPResamplerModule(image_encoder, resampler)
    clip_resampler.eval()
    clip_resampler.to(dtype)

    # Save to cache for future use
    if cache_path:
        save_pt_model(clip_resampler, cache_path)

    return clip_resampler


def run_pytorch_model(input):
    model = get_pt_model()

    with torch.inference_mode():
        outputs = model(**input)

    return outputs


def main():
    """Main function to run the TTNN model."""
    # Load input tensor
    input_torch = get_input()

    # Calculate torch output
    output_torch = run_pytorch_model(input_torch)

    # Convert torch input to host TTNN tensor
    input_ttnn_host = ttnn.from_torch(input_torch["pixel_values"])
    input_ttnn_host = ttnn.to_layout(input_ttnn_host, ttnn.Layout.TILE)
    input_ttnn_host = ttnn.to_dtype(input_ttnn_host, ttnn.DataType.BFLOAT16)

    # Load weights from PyTorch state_dict
    print("Loading weights from PyTorch state_dict...")
    pt_model = get_pt_model()
    state_dict = pt_model.state_dict()
    weights = load_inputs_from_pytorch(state_dict)

    # Create TTNN model
    model = CLIPResampler(weights)

    # Get device
    device = utils.DeviceGetter.get_device((1, 1))

    # Run ttnn model
    for i in range(3):
        start_time = time.time()

        # Move input to device
        pixel_values = ttnn.to_device(
            input_ttnn_host,
            device,
            ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
            ),
        )

        # Run ttnn model
        out_ttnn_device = model(pixel_values)[0]

        # Get outputs
        out_ttnn_host = ttnn.from_device(out_ttnn_device, blocking=True)
        ttnn.synchronize_device(device)

        end_time = time.time()

        # Calculate FPS and PCC
        duration = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time)  # batch size is 1
        pcc = utils.calculate_pcc(output_torch, ttnn.to_torch(out_ttnn_host))

        # Print results
        print(f"Iteration {i}")
        print(f"\tDuration: {duration:.1f}ms")
        print(f"\tFPS: {fps:.2f}")
        print(f"\tPCC: {pcc:.6f}")

    return 0


if __name__ == "__main__":
    main()
