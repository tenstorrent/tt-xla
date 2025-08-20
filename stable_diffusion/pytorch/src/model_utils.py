# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion v3.5 model loading and processing.
"""

from typing import List, Optional, Tuple, Union
import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


def load_pipe(variant):
    """Load Stable Diffusion v3.5 pipeline.

    Args:
        variant: Model variant name

    Returns:
        StableDiffusion3Pipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusion3Pipeline.from_pretrained(
        f"stabilityai/{variant}", torch_dtype=torch.float32
    )
    modules = [
        pipe.text_encoder,
        pipe.transformer,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
        pipe.vae,
    ]

    # Move the pipeline to CPU
    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def calculate_shift(
    image_seq_len,
    base_image_seq_len,
    max_image_seq_len,
    base_shift,
    max_shift,
):
    """Calculate dynamic shifting parameter for the scheduler.

    Args:
        image_seq_len: Current image sequence length
        base_image_seq_len: Base image sequence length
        max_image_seq_len: Maximum image sequence length
        base_shift: Base shift value
        max_shift: Maximum shift value

    Returns:
        float: Calculated shift parameter
    """
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = image_seq_len * m + b
    return mu


def stable_diffusion_preprocessing_v35(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.0,
    num_inference_steps=1,
    num_images_per_prompt=1,
    clip_skip=None,
    max_sequence_length=256,
    joint_attention_kwargs=None,
    skip_guidance_layers=None,
    skip_layer_guidance_scale=2.8,
    skip_layer_guidance_start=0.01,
    skip_layer_guidance_stop=0.2,
    do_classifier_free_guidance=True,
    mu=None,
):
    """Preprocess inputs for Stable Diffusion v3.5 model.

    Args:
        pipe: Stable Diffusion v3.5 pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 7.0)
        num_inference_steps: Number of inference steps (default: 1)
        num_images_per_prompt: Number of images per prompt (default: 1)
        clip_skip: CLIP skip layers (optional)
        max_sequence_length: Maximum sequence length (default: 256)
        joint_attention_kwargs: Joint attention kwargs (optional)
        skip_guidance_layers: Skip guidance layers (optional)
        skip_layer_guidance_scale: Skip layer guidance scale (default: 2.8)
        skip_layer_guidance_start: Skip layer guidance start (default: 0.01)
        skip_layer_guidance_stop: Skip layer guidance stop (default: 0.2)
        do_classifier_free_guidance: Whether to use classifier-free guidance (default: True)
        mu: Dynamic shifting parameter (optional)

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds)
    """
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor

    pipe.check_inputs(
        prompt,
        None,  # prompt_2
        None,  # prompt_3
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        device=device,
        clip_skip=clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )

    if do_classifier_free_guidance:
        original_prompt_embeds = prompt_embeds
        original_pooled_prompt_embeds = pooled_prompt_embeds

        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        num_images_per_prompt,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)

    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        image_seq_len = (height // pipe.transformer.config.patch_size) * (
            width // pipe.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=1,
        device=device,
        sigmas=None,
        **scheduler_kwargs,
    )

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds
