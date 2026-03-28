# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for SDXL InstructPix2Pix model loading and processing.
"""

from typing import Optional, Tuple

import torch
from diffusers import StableDiffusionXLInstructPix2PixPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from diffusers.utils import retrieve_latents
from PIL import Image


def load_sdxl_instructpix2pix_pipe(pretrained_model_name):
    """Load SDXL InstructPix2Pix pipeline.

    Args:
        pretrained_model_name: Model name on HuggingFace

    Returns:
        StableDiffusionXLInstructPix2PixPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32
    )

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def create_dummy_input_image(height=768, width=768):
    """Create a dummy input image for editing.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy input image
    """
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def sdxl_instructpix2pix_preprocessing(
    pipe,
    prompt,
    image,
    device="cpu",
    negative_prompt=None,
    guidance_scale=3.0,
    image_guidance_scale=1.5,
    num_inference_steps=30,
    timesteps=None,
    sigmas=None,
    num_images_per_prompt=1,
    height=None,
    width=None,
    clip_skip=None,
    original_size=None,
    target_size=None,
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_target_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
):
    """Preprocess inputs for SDXL InstructPix2Pix model.

    Args:
        pipe: StableDiffusionXLInstructPix2PixPipeline
        prompt: Edit instruction text
        image: Input image to edit
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Text guidance scale (default: 3.0)
        image_guidance_scale: Image guidance scale (default: 1.5)
        num_inference_steps: Number of inference steps (default: 30)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional)
        width: Image width (optional)
        clip_skip: CLIP skip layers (optional)
        original_size: Original size tuple (optional)
        target_size: Target size tuple (optional)
        crops_coords_top_left: Crop coordinates (default: (0, 0))
        negative_original_size: Negative original size (optional)
        negative_target_size: Negative target size (optional)
        negative_crops_coords_top_left: Negative crop coordinates (default: (0, 0))

    Returns:
        tuple: (scaled_latent_model_input, timesteps, prompt_embeds, added_cond_kwargs)
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

    # 3. Prepare noise latents
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    num_channels_latents = pipe.vae.config.latent_channels
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

    # 4. Prepare image latents by encoding the input image through VAE
    image_tensor = pipe.image_processor.preprocess(image, height=height, width=width)
    image_tensor = image_tensor.to(device=device, dtype=pipe.vae.dtype)
    image_latents = retrieve_latents(
        pipe.vae.encode(image_tensor), sample_mode="argmax"
    )

    # Duplicate image latents for classifier-free guidance:
    # [image_latents, image_latents, zeros] for [text, image, uncond] branches
    uncond_image_latents = torch.zeros_like(image_latents)
    image_latents = torch.cat(
        [image_latents, image_latents, uncond_image_latents], dim=0
    )

    # 5. Prepare additional conditioning
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

    # Triple for classifier-free guidance: [text, image, uncond]
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat(
            [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds], dim=0
        )
        add_text_embeds = torch.cat(
            [
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ],
            dim=0,
        )
        add_time_ids = torch.cat(
            [add_time_ids, negative_add_time_ids, negative_add_time_ids], dim=0
        )

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    # 6. Prepare latent model input (tripled for CFG) and concatenate with image latents
    latent_model_input = torch.cat([latents] * 3)
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )
    scaled_latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)

    return (
        scaled_latent_model_input,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
    )
