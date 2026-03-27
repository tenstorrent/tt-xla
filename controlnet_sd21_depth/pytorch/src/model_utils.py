# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet SD2.1 Depth model loading and processing.
"""

import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image


def load_controlnet_sd21_depth_pipe(controlnet_model_name, base_model_name):
    """Load ControlNet SD2.1 Depth pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base SD2.1 model name on HuggingFace

    Returns:
        StableDiffusionControlNetPipeline: Loaded pipeline with components set to eval mode
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_name, torch_dtype=torch.float32
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_model_name, controlnet=controlnet, torch_dtype=torch.float32
    )

    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.unet,
        pipe.vae,
        pipe.controlnet,
    ]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def create_depth_conditioning_image(height=768, width=768):
    """Create a dummy depth conditioning image.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy depth conditioning image
    """
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def controlnet_sd21_depth_preprocessing(
    pipe,
    prompt,
    control_image,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.5,
    num_inference_steps=50,
    timesteps=None,
    sigmas=None,
    num_images_per_prompt=1,
    height=None,
    width=None,
    controlnet_conditioning_scale=1.0,
):
    """Preprocess inputs for ControlNet SD2.1 Depth model.

    Args:
        pipe: StableDiffusionControlNetPipeline
        prompt: Text prompt for generation
        control_image: Depth conditioning image
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 7.5)
        num_inference_steps: Number of inference steps (default: 50)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional)
        width: Image width (optional)
        controlnet_conditioning_scale: ControlNet conditioning scale (default: 1.0)

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds,
                down_block_additional_residuals, mid_block_additional_residual)
    """
    default_sample_size = pipe.unet.config.sample_size
    height = height or default_sample_size * pipe.vae_scale_factor
    width = width or default_sample_size * pipe.vae_scale_factor

    do_classifier_free_guidance = True

    # 1. Encode the prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
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

    # 4. Prepare classifier-free guidance
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    prompt_embeds = prompt_embeds.to(device)

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
    down_block_additional_residuals, mid_block_additional_residual = pipe.controlnet(
        latent_model_input,
        timesteps[0],
        encoder_hidden_states=prompt_embeds,
        controlnet_cond=control_image,
        conditioning_scale=controlnet_conditioning_scale,
        return_dict=False,
    )

    return (
        latent_model_input,
        timesteps,
        prompt_embeds,
        down_block_additional_residuals,
        mid_block_additional_residual,
    )
