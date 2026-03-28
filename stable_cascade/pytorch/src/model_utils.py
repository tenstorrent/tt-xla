# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Cascade model loading and processing.
"""

import torch
from diffusers import StableCascadePriorPipeline


def load_prior_pipe(variant):
    """Load Stable Cascade prior pipeline.

    Args:
        variant: Pretrained model name for the prior

    Returns:
        StableCascadePriorPipeline: Loaded prior pipeline with components set to eval mode
    """
    pipe = StableCascadePriorPipeline.from_pretrained(
        variant, torch_dtype=torch.float32
    )

    pipe.to("cpu")

    for module in [pipe.prior, pipe.text_encoder]:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def stable_cascade_preprocessing(
    prior_pipe, prompt, device="cpu", num_inference_steps=20
):
    """Preprocess inputs for the Stable Cascade prior model.

    Args:
        prior_pipe: Stable Cascade prior pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 20)

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds)
    """
    batch_size = 1
    height = 1024
    width = 1024

    # Encode prompt
    (
        prompt_embeds,
        prompt_embeds_pooled,
        negative_prompt_embeds,
        negative_prompt_embeds_pooled,
    ) = prior_pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    # Prepare timesteps
    prior_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = prior_pipe.scheduler.timesteps

    # Prepare latent variables - the prior operates at a highly compressed resolution
    latent_height = height // prior_pipe.config.resolution_multiple
    latent_width = width // prior_pipe.config.resolution_multiple
    num_channels = prior_pipe.prior.config.in_channels

    torch.manual_seed(42)
    latents = torch.randn(
        (batch_size, num_channels, latent_height, latent_width),
        device=device,
    )

    # Concatenate for classifier-free guidance
    latent_model_input = torch.cat([latents] * 2)
    timestep = timesteps[:1]

    prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])

    return (latent_model_input, timestep, prompt_embeds_input)
