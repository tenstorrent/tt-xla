# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for CogVideoX model loading and processing.
"""

import torch
from diffusers import CogVideoXPipeline


def load_pipe(model_name):
    """Load CogVideoX pipeline.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        CogVideoXPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = CogVideoXPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

    modules = [
        pipe.text_encoder,
        pipe.transformer,
        pipe.vae,
    ]

    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def cogvideox_preprocessing(pipe, prompt, device="cpu", num_inference_steps=1):
    """Preprocess inputs for CogVideoX transformer model.

    Args:
        pipe: CogVideoX pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 1)

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds)
    """
    # Encode prompt using the pipeline's text encoder
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        device=device,
        dtype=torch.float32,
    )

    # CogVideoX-5b: transformer expects latents of shape
    # (batch, num_frames, channels, height, width)
    # Default: 49 frames, 16 latent channels, latent spatial dims
    num_channels_latents = pipe.transformer.config.in_channels
    num_frames = 13  # (49 video frames - 1) / temporal_compression_ratio(4) + 1
    latent_height = 60  # 480 / vae_scale_factor(8)
    latent_width = 90  # 720 / vae_scale_factor(8)

    latents = torch.randn(
        (1, num_frames, num_channels_latents, latent_height, latent_width),
        device=device,
        dtype=torch.float32,
    )

    # Set up scheduler and get timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    timestep = timesteps[0].unsqueeze(0)

    return latents, timestep, prompt_embeds
