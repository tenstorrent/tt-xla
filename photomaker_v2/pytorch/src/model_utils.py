# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for PhotoMaker V2 model loading and processing.
"""

import os
from typing import Optional, Tuple
import torch
from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download
from photomaker import PhotoMakerStableDiffusionXLPipeline
from PIL import Image
import numpy as np


def load_photomaker_v2_pipe(base_model_name, photomaker_repo_id):
    """Load PhotoMaker V2 pipeline.

    Args:
        base_model_name: Base SDXL model name on HuggingFace
        photomaker_repo_id: PhotoMaker V2 model repo ID on HuggingFace

    Returns:
        PhotoMakerStableDiffusionXLPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        base_model_name, torch_dtype=torch.float32
    )

    photomaker_path = hf_hub_download(
        repo_id=photomaker_repo_id,
        filename="photomaker-v2.bin",
        repo_type="model",
    )

    pipe.load_photomaker_adapter(
        os.path.dirname(photomaker_path),
        subfolder="",
        weight_name=os.path.basename(photomaker_path),
        trigger_word="img",
    )
    pipe.fuse_lora()

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def create_dummy_id_image(height=256, width=256):
    """Create a dummy identity image for PhotoMaker V2.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy face image
    """
    np.random.seed(42)
    data = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data)


def photomaker_v2_preprocessing(
    pipe,
    prompt,
    input_id_images,
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
    crops_coords_top_left: Tuple[int, int] = (0, 0),
    negative_original_size: Optional[Tuple[int, int]] = None,
    negative_target_size: Optional[Tuple[int, int]] = None,
    negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
):
    """Preprocess inputs for PhotoMaker V2 model.

    Args:
        pipe: PhotoMakerStableDiffusionXLPipeline
        prompt: Text prompt for generation (must contain trigger word "img")
        input_id_images: List of PIL images for identity extraction
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
        crops_coords_top_left: Crop coordinates (default: (0, 0))
        negative_original_size: Negative original size (optional)
        negative_target_size: Negative target size (optional)
        negative_crops_coords_top_left: Negative crop coordinates (default: (0, 0))

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds, added_cond_kwargs)
    """
    default_sample_size = pipe.unet.config.sample_size
    height = height or default_sample_size * pipe.vae_scale_factor
    width = width or default_sample_size * pipe.vae_scale_factor
    original_size = original_size or (height, width)
    target_size = target_size or (height, width)

    do_classifier_free_guidance = True

    # 1. Encode the prompt with PhotoMaker identity encoding
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt_with_trigger_word(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        clip_skip=clip_skip,
        input_id_images=input_id_images,
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

    # 5. Prepare latent model input
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )

    return (
        latent_model_input,
        timesteps,
        prompt_embeds,
        added_cond_kwargs,
    )
