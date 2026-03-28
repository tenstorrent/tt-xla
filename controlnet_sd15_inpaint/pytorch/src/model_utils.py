# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet SD1.5 Inpaint model loading and processing.
"""

import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image


def load_controlnet_sd15_inpaint_pipe(controlnet_model_name, base_model_name):
    """Load ControlNet SD1.5 Inpaint pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base SD1.5 model name on HuggingFace

    Returns:
        StableDiffusionControlNetInpaintPipeline: Loaded pipeline with components set to eval mode
    """
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_name, torch_dtype=torch.float32
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_name, controlnet=controlnet, torch_dtype=torch.float32
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

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


def make_inpaint_condition(image, image_mask):
    """Create an inpaint conditioning tensor from an image and mask.

    Args:
        image: PIL Image (RGB)
        image_mask: PIL Image (grayscale mask, white = masked region)

    Returns:
        torch.Tensor: Conditioning tensor of shape (1, 3, H, W)
    """
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0
    image[image_mask > 0.5] = -1.0
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def create_dummy_inpaint_images(height=512, width=512):
    """Create dummy input image and mask for inpainting.

    Args:
        height: Image height
        width: Image width

    Returns:
        tuple: (init_image, mask_image) as PIL Images
    """
    init_image = Image.new("RGB", (width, height), color=(128, 128, 128))
    mask_image = Image.new("L", (width, height), color=0)
    # Create a white square in the center as the mask region
    from PIL import ImageDraw

    draw = ImageDraw.Draw(mask_image)
    draw.rectangle([width // 4, height // 4, 3 * width // 4, 3 * height // 4], fill=255)
    return init_image, mask_image


def controlnet_sd15_inpaint_preprocessing(
    pipe,
    prompt,
    init_image,
    mask_image,
    control_image,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.5,
    num_inference_steps=20,
    timesteps=None,
    sigmas=None,
    num_images_per_prompt=1,
    height=512,
    width=512,
    controlnet_conditioning_scale=1.0,
    eta=1.0,
):
    """Preprocess inputs for ControlNet SD1.5 Inpaint model.

    Args:
        pipe: StableDiffusionControlNetInpaintPipeline
        prompt: Text prompt for generation
        init_image: Original image for inpainting
        mask_image: Mask image indicating region to inpaint
        control_image: Inpaint conditioning tensor
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 7.5)
        num_inference_steps: Number of inference steps (default: 20)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (default: 512)
        width: Image width (default: 512)
        controlnet_conditioning_scale: ControlNet conditioning scale (default: 1.0)
        eta: Eta parameter for DDIM scheduler (default: 1.0)

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds,
                down_block_additional_residuals, mid_block_additional_residual)
    """
    do_classifier_free_guidance = True
    batch_size = 1 if isinstance(prompt, str) else len(prompt)

    # 1. Encode the prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    prompt_embeds = prompt_embeds.to(device)

    # 2. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        timesteps=timesteps,
        sigmas=sigmas,
    )

    # 3. Prepare latent variables
    num_channels_latents = (
        4  # noise latent channels only, not full in_channels (9 for inpaint UNet)
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

    # 4. Prepare control image for controlnet
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

    # 5. Prepare latent model input
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    latent_model_input = pipe.scheduler.scale_model_input(
        latent_model_input, timesteps[0]
    )

    # 6. Run controlnet to get residuals
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
