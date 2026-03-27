# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for InstructPix2Pix model loading and processing.
"""

import torch
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    EulerAncestralDiscreteScheduler,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image


def load_instruct_pix2pix_pipe(model_name):
    """Load InstructPix2Pix pipeline.

    Args:
        model_name: Model name on HuggingFace

    Returns:
        StableDiffusionInstructPix2PixPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
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
    """Create a dummy input image for InstructPix2Pix.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy input image
    """
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def instruct_pix2pix_preprocessing(
    pipe,
    prompt,
    image,
    device="cpu",
    num_inference_steps=10,
    guidance_scale=7.5,
    image_guidance_scale=1.5,
    num_images_per_prompt=1,
):
    """Preprocess inputs for InstructPix2Pix model.

    Args:
        pipe: StableDiffusionInstructPix2PixPipeline
        prompt: Text instruction for image editing
        image: Input image to edit
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 10)
        guidance_scale: Text guidance scale (default: 7.5)
        image_guidance_scale: Image guidance scale (default: 1.5)
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

    # For InstructPix2Pix, we need three copies for classifier-free guidance:
    # [negative, image-only (no text), full (text+image)]
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat(
            [negative_prompt_embeds, negative_prompt_embeds, prompt_embeds]
        )

    # 2. Prepare image latents
    image = pipe.image_processor.preprocess(image, height=height, width=width)
    image_latents = (
        pipe.vae.encode(image).latent_dist.mode() * pipe.vae.config.scaling_factor
    )

    if do_classifier_free_guidance:
        image_latents = torch.cat([image_latents, image_latents, image_latents])

    # 3. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
    )

    # 4. Prepare latent variables
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    num_channels_latents = pipe.unet.config.in_channels // 2
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

    # 5. Prepare latent model input (concat noise latents with image latents)
    scaled_latent_model_input = pipe.scheduler.scale_model_input(latents, timesteps[0])
    if do_classifier_free_guidance:
        scaled_latent_model_input = torch.cat([scaled_latent_model_input] * 3)

    # Concatenate image latents along the channel dimension
    latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds
