# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for MV-Adapter model loading and preprocessing.
"""

import torch
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)

from mvadapter.pipelines.pipeline_mvadapter_t2mv_sdxl import MVAdapterT2MVSDXLPipeline
from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler


NUM_VIEWS = 6
HEIGHT = 768
WIDTH = 768


def load_mv_adapter_pipeline(adapter_path, base_model):
    """Load the MV-Adapter T2MV SDXL pipeline.

    Args:
        adapter_path: HuggingFace path for the MV-Adapter weights.
        base_model: Base SDXL model name on HuggingFace.

    Returns:
        MVAdapterT2MVSDXLPipeline: The loaded pipeline with components in eval mode.
    """
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32
    )
    pipe = MVAdapterT2MVSDXLPipeline.from_pretrained(
        base_model, vae=vae, torch_dtype=torch.float32
    )

    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
    )

    pipe.init_custom_adapter(num_views=NUM_VIEWS)
    pipe.load_custom_adapter(
        adapter_path, weight_name="mvadapter_t2mv_sdxl.safetensors"
    )

    pipe.to("cpu")

    modules = [pipe.text_encoder, pipe.unet, pipe.text_encoder_2, pipe.vae]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def mv_adapter_preprocessing(
    pipe,
    prompt,
    device="cpu",
    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
    guidance_scale=7.0,
    num_inference_steps=50,
):
    """Preprocess inputs for the MV-Adapter UNet.

    Args:
        pipe: MVAdapterT2MVSDXLPipeline instance.
        prompt: Text prompt for generation.
        device: Device to run on.
        negative_prompt: Negative prompt for classifier-free guidance.
        guidance_scale: Guidance scale.
        num_inference_steps: Number of inference steps.

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds, added_cond_kwargs)
    """
    num_views = NUM_VIEWS
    height = HEIGHT
    width = WIDTH

    do_classifier_free_guidance = guidance_scale > 1.0

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
        num_images_per_prompt=num_views,
    )

    # 2. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    # 3. Prepare latent variables
    num_channels_latents = pipe.unet.config.in_channels
    torch.manual_seed(42)
    latents = torch.randn(
        (
            num_views,
            num_channels_latents,
            height // pipe.vae_scale_factor,
            width // pipe.vae_scale_factor,
        ),
        device=device,
    )
    latents = latents * pipe.scheduler.init_noise_sigma

    # 4. Prepare additional conditioning (SDXL time ids)
    original_size = (height, width)
    target_size = (height, width)
    crops_coords_top_left = (0, 0)

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

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat(
            [negative_pooled_prompt_embeds, add_text_embeds], dim=0
        )
        negative_add_time_ids = add_time_ids
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(num_views, 1)

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
