# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Anime Detailer XL LoRA model loading and processing.
"""

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(base_model_name, lora_model_id, lora_filename, lora_scale=2):
    """Load Stable Diffusion XL pipeline with LoRA weights applied.

    Args:
        base_model_name: Base SDXL model name
        lora_model_id: HuggingFace LoRA model ID
        lora_filename: LoRA weights filename
        lora_scale: LoRA scale factor (default: 2)

    Returns:
        StableDiffusionXLPipeline: Loaded pipeline with LoRA weights fused
    """
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        use_safetensors=True,
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cpu")

    # Load and fuse LoRA weights
    pipe.load_lora_weights(lora_model_id, weight_name=lora_filename)
    pipe.fuse_lora(lora_scale=lora_scale)

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe
