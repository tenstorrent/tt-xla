# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for LCM-LoRA SDXL model loading and processing.
"""

from typing import Optional, Tuple

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


BASE_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


def load_pipe(adapter_id, base_model_id=BASE_MODEL_ID):
    """Load SDXL pipeline with LCM-LoRA weights fused in.

    Args:
        adapter_id: HuggingFace model ID for the LCM-LoRA adapter.
        base_model_id: HuggingFace model ID for the base SDXL model.

    Returns:
        AutoPipelineForText2Image: Pipeline with LCM-LoRA fused and LCMScheduler.
    """
    pipe = AutoPipelineForText2Image.from_pretrained(
        base_model_id, torch_dtype=torch.float32
    )

    # Replace scheduler with LCMScheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # Load and fuse LCM-LoRA weights
    pipe.load_lora_weights(adapter_id)
    pipe.fuse_lora()

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def lcm_lora_sdxl_preprocessing(
    pipe,
    prompt,
    device="cpu",
    num_inference_steps=4,
    num_images_per_prompt=1,
    height=None,
    width=None,
    original_size=None,
    target_size=None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
):
    """Preprocess inputs for LCM-LoRA SDXL model.

    LCM uses guidance_scale=0, so classifier-free guidance is disabled.

    Args:
        pipe: LCM-LoRA SDXL pipeline.
        prompt: Text prompt for generation.
        device: Device to run on (default: "cpu").
        num_inference_steps: Number of inference steps (default: 4).
        num_images_per_prompt: Number of images per prompt (default: 1).
        height: Image height (optional).
        width: Image width (optional).
        original_size: Original size tuple (optional).
        target_size: Target size tuple (optional).
        crops_coords_top_left: Crop coordinates (default: (0, 0)).

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds, timestep_cond,
                added_cond_kwargs, add_time_ids)
    """
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # LCM does not use classifier-free guidance
    do_classifier_free_guidance = False

    # Encode prompt (no negative prompt needed without CFG)
    (
        prompt_embeds,
        _negative_prompt_embeds,
        pooled_prompt_embeds,
        _negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
    )

    # Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    # Prepare latent variables
    if isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    num_channels_latents = pipe.unet.config.in_channels
    torch.manual_seed(42)
    latents = torch.randn(
        (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    # Prepare added conditioning kwargs
    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(
        original_size,
        crops_coords_top_left,
        target_size,
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    # Timestep conditioning
    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(0.0).repeat(
            batch_size * num_images_per_prompt
        )
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor,
            embedding_dim=pipe.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # No CFG doubling needed
    latent_model_input = pipe.scheduler.scale_model_input(latents, timesteps[0])

    return (
        latent_model_input,
        timesteps,
        prompt_embeds,
        timestep_cond,
        added_cond_kwargs,
        add_time_ids,
    )
