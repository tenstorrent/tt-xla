# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for WAMU I2V Lightning model loading."""

import torch
from PIL import Image


def load_i2v_pipeline(pretrained_model_name: str, dtype: torch.dtype):
    """
    Load WanImageToVideoPipeline from diffusers.

    The image encoder and VAE are loaded in float32 for numerical stability,
    while the main transformer uses the provided dtype.

    Args:
        pretrained_model_name: HuggingFace model ID
        dtype: Torch dtype for the transformer weights
    """
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel

    image_encoder = CLIPVisionModel.from_pretrained(
        pretrained_model_name,
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        pretrained_model_name,
        vae=vae,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )
    return pipe


def load_i2v_inputs(prompt: str) -> dict:
    """
    Prepare inputs for the I2V pipeline (image-to-video generation).

    Returns a dict suitable for passing to WanImageToVideoPipeline.__call__.
    Uses a small synthetic image for testing.
    """
    ref_image = Image.new("RGB", (832, 480), color=(128, 128, 200))

    return {
        "image": ref_image,
        "prompt": prompt,
        "height": 480,
        "width": 832,
        "num_frames": 9,
        "num_inference_steps": 2,
        "guidance_scale": 5.0,
    }
