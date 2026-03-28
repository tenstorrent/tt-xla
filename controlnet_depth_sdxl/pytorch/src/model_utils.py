# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet Depth SDXL model loading and processing.
"""

from typing import Optional, Tuple
import torch
from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image


def load_controlnet_depth_sdxl_pipe(controlnet_model_name, base_model_name):
    """Load ControlNet Depth SDXL pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base SDXL model name on HuggingFace

    Returns:
        StableDiffusionXLControlNetPipeline: Loaded pipeline with components set to eval mode
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_name, torch_dtype=torch.float32
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_name, controlnet=controlnet, torch_dtype=torch.float32
    )

    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.unet,
        pipe.text_encoder_2,
        pipe.vae,
        pipe.controlnet,
    ]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def create_depth_conditioning_image(height=1024, width=1024):
    """Create a dummy depth conditioning image.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy depth conditioning image
    """
    return Image.new("RGB", (width, height), color=(0, 0, 0))


def controlnet_depth_sdxl_preprocessing(
    pipe,
    prompt,
    control_image,
    device="cpu",
    negative_prompt=None,
    guidance_scale=5.0,
    num_inference_steps=50,
    timesteps=None,
    sigmas=None,
    num_images_per_prompt=1,
    height=None,
    width=None,
    clip_skip=None,
    original_size=None,
    target_size=None,
    controlnet_conditioning_scale=1.0,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_target_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
):
    """Preprocess inputs for ControlNet Depth SDXL model.

    Args:
        pipe: StableDiffusionXLControlNetPipeline
        prompt: Text prompt for generation
        control_image: Depth conditioning image
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 5.0)
        num_inference_steps: Number of inference steps (default: 50)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional)
        width: Image width (optional)
        clip_skip: CLIP skip layers (optional)
        original_size: Original size tuple (optional)
        target_size: Target size tuple (optional)
        controlnet_conditioning_scale: ControlNet conditioning scale (default: 1.0)
        crops_coords_top_left: Crop coordinates (default: (0, 0))
        negative_original_size: Negative original size (optional)
        negative_target_size: Negative target size (optional)
        negative_crops_coords_top_left: Negative crop coordinates (default: (0, 0))

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds, added_cond_kwargs,
                down_block_additional_residuals, mid_block_additional_residual)
    """
    default_sample_size = pipe.unet.config.sample_size
    height = height or default_sample_size * pipe.vae_scale_factor
    width = width or default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    do_classifier_free_guidance = True

    # 1. Encode the prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
    )

    # 2. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
    )

    # 3. Prepare latent variables
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
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

    # 4. Prepare additional conditioning
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
    if negative_original_size is not None and negative_target_size is not None:
        negative_add_time_ids = pipe._get_add_time_ids(
            negative_original_size,
            negative_crops_coords_top_left,
            negative_target_size,
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        )
    else:
        negative_add_time_ids = add_time_ids

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # 5. Prepare control image
    control_image = pipe.prepare_image(
        image=control_image,
        width=width,
        height=height,
        batch_size=batch_size * num_images_per_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        dtype=pipe.controlnet.dtype,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # 6. Prepare latent model input
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )

    # 7. Run controlnet to get residuals
    controlnet_cond_scale = controlnet_conditioning_scale
    down_block_additional_residuals, mid_block_additional_residual = pipe.controlnet(
        latent_model_input,
        timesteps[0],
        encoder_hidden_states=prompt_embeds,
        controlnet_cond=control_image,
        conditioning_scale=controlnet_cond_scale,
        added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )

    return (
        latent_model_input,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
        down_block_additional_residuals,
        mid_block_additional_residual,
    )
