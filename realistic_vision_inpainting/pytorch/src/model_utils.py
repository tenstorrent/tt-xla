# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Realistic Vision V1.4 Inpainting model loading and processing.
"""

import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image, ImageDraw


def load_inpainting_pipe(model_name):
    """Load Stable Diffusion Inpainting pipeline.

    Args:
        model_name: Model name on HuggingFace

    Returns:
        StableDiffusionInpaintPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name, torch_dtype=torch.float32, safety_checker=None
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.unet,
        pipe.vae,
    ]
    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def create_dummy_input_image(height=512, width=512):
    """Create a dummy input image for inpainting.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy input image
    """
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def create_dummy_mask_image(height=512, width=512):
    """Create a dummy mask image for inpainting.

    The mask is white (255) in the center region to indicate the area to inpaint.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy mask image
    """
    mask = Image.new("L", (width, height), color=0)
    # Create a white square in the center (region to inpaint)
    draw = ImageDraw.Draw(mask)
    center_x, center_y = width // 2, height // 2
    box_size = min(width, height) // 4
    draw.rectangle(
        [
            center_x - box_size,
            center_y - box_size,
            center_x + box_size,
            center_y + box_size,
        ],
        fill=255,
    )
    return mask


def inpainting_preprocessing(
    pipe,
    prompt,
    image,
    mask_image,
    device="cpu",
    num_inference_steps=10,
    guidance_scale=7.5,
    num_images_per_prompt=1,
):
    """Preprocess inputs for Stable Diffusion Inpainting model.

    Args:
        pipe: StableDiffusionInpaintPipeline
        prompt: Text prompt for inpainting
        image: Input image
        mask_image: Mask image indicating region to inpaint
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 10)
        guidance_scale: Text guidance scale (default: 7.5)
        num_images_per_prompt: Number of images per prompt (default: 1)

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds)
    """
    height = width = pipe.unet.config.sample_size * pipe.vae_scale_factor

    do_classifier_free_guidance = guidance_scale > 1.0

    # 1. Encode the prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 2. Prepare image and mask latents
    image = pipe.image_processor.preprocess(image, height=height, width=width)
    mask = pipe.mask_processor.preprocess(mask_image, height=height, width=width)

    masked_image = image * (mask < 0.5)
    masked_image_latents = (
        pipe.vae.encode(masked_image).latent_dist.mode()
        * pipe.vae.config.scaling_factor
    )

    # Resize mask to latent space
    mask = torch.nn.functional.interpolate(
        mask,
        size=(height // pipe.vae_scale_factor, width // pipe.vae_scale_factor),
    )

    if do_classifier_free_guidance:
        masked_image_latents = torch.cat([masked_image_latents] * 2)
        mask = torch.cat([mask] * 2)

    # 3. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    # 4. Prepare latent variables
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    num_channels_latents = (
        pipe.unet.config.in_channels - mask.shape[1] - masked_image_latents.shape[1]
    )
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

    # 5. Prepare latent model input (concat noise latents, mask, and masked image latents)
    scaled_latent_model_input = pipe.scheduler.scale_model_input(latents, timesteps[0])
    if do_classifier_free_guidance:
        scaled_latent_model_input = torch.cat([scaled_latent_model_input] * 2)

    latent_model_input = torch.cat(
        [scaled_latent_model_input, mask, masked_image_latents], dim=1
    )

    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds
