# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for DMD2 model loading and processing.
"""

from typing import Tuple

import torch
from diffusers import DiffusionPipeline, LCMScheduler, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download


def load_pipe(base_model_id, repo_name, ckpt_name):
    """Load DMD2 pipeline with distilled UNet weights.

    Args:
        base_model_id: Base SDXL model identifier
        repo_name: HuggingFace repo containing DMD2 weights
        ckpt_name: Checkpoint filename to download

    Returns:
        DiffusionPipeline: Loaded pipeline with DMD2 UNet and LCMScheduler
    """
    # Load the base UNet config and apply distilled weights
    unet = UNet2DConditionModel.from_config(base_model_id, subfolder="unet").to(
        torch.float32
    )
    unet.load_state_dict(
        torch.load(
            hf_hub_download(repo_name, ckpt_name),
            map_location="cpu",
            weights_only=True,
        )
    )

    # Load the base SDXL pipeline with the distilled UNet
    pipe = DiffusionPipeline.from_pretrained(
        base_model_id, unet=unet, torch_dtype=torch.float32
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]

    # Move the pipeline to CPU
    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def dmd2_preprocessing(
    pipe,
    prompt,
    device="cpu",
    num_inference_steps=4,
    timesteps_list=None,
    num_images_per_prompt=1,
    height=None,
    width=None,
    original_size=None,
    target_size=None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
):
    """Preprocess inputs for DMD2 model.

    DMD2 uses guidance_scale=0 (no classifier-free guidance) since guidance
    was distilled into the model during training.

    Args:
        pipe: DMD2 pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 4)
        timesteps_list: Custom timesteps list (default: [999, 749, 499, 249])
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional, uses default if None)
        width: Image width (optional, uses default if None)
        original_size: Original size tuple (optional)
        target_size: Target size tuple (optional)
        crops_coords_top_left: Crop coordinates (default: (0, 0))

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, added_cond_kwargs)
    """
    if timesteps_list is None:
        timesteps_list = [999, 749, 499, 249]

    # Set default height and width
    height = height or pipe.default_sample_size * pipe.vae_scale_factor
    width = width or pipe.default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    # DMD2 does not use classifier-free guidance
    do_classifier_free_guidance = False

    # Encode the prompt (no negative prompt needed)
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
        timesteps=timesteps_list,
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

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    latent_model_input = pipe.scheduler.scale_model_input(latents, timesteps[0])

    return (
        latent_model_input,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
    )
