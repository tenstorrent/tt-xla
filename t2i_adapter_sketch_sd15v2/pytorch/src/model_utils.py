# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for T2I-Adapter Sketch SD1.5v2 model loading and processing.
"""

import torch
from diffusers import StableDiffusionAdapterPipeline, T2IAdapter
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from diffusers.pipelines.t2i_adapter.pipeline_stable_diffusion_adapter import (
    _preprocess_adapter_image,
)
from PIL import Image


def load_t2i_adapter_sketch_sd15v2_pipe(adapter_model_name, base_model_name):
    """Load T2I-Adapter Sketch SD1.5v2 pipeline.

    Args:
        adapter_model_name: T2I-Adapter model name on HuggingFace
        base_model_name: Base SD1.5 model name on HuggingFace

    Returns:
        StableDiffusionAdapterPipeline: Loaded pipeline with components set to eval mode
    """
    adapter = T2IAdapter.from_pretrained(adapter_model_name, torch_dtype=torch.float32)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        base_model_name, adapter=adapter, torch_dtype=torch.float32
    )

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


def create_sketch_conditioning_image(height=512, width=512):
    """Create a dummy sketch conditioning image.

    Args:
        height: Image height
        width: Image width

    Returns:
        PIL.Image: A dummy conditioning image
    """
    return Image.new("RGB", (width, height), color=(255, 255, 255))


def t2i_adapter_sketch_sd15v2_preprocessing(
    pipe,
    prompt,
    adapter_image,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.5,
    num_inference_steps=50,
    timesteps=None,
    sigmas=None,
    num_images_per_prompt=1,
    height=None,
    width=None,
    adapter_conditioning_scale=0.9,
):
    """Preprocess inputs for T2I-Adapter Sketch SD1.5v2 model.

    Args:
        pipe: StableDiffusionAdapterPipeline
        prompt: Text prompt for generation
        adapter_image: Sketch conditioning image
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 7.5)
        num_inference_steps: Number of inference steps (default: 50)
        timesteps: Custom timesteps (optional)
        sigmas: Custom sigmas (optional)
        num_images_per_prompt: Number of images per prompt (default: 1)
        height: Image height (optional)
        width: Image width (optional)
        adapter_conditioning_scale: Adapter conditioning scale (default: 0.9)

    Returns:
        tuple: (latent_model_input, timesteps, prompt_embeds,
                down_intrablock_additional_residuals)
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

    # 4. Prepare adapter image and get adapter features
    adapter_input = _preprocess_adapter_image(adapter_image, height, width)
    adapter_input = adapter_input.to(device=device, dtype=pipe.adapter.dtype)

    # Run adapter to get down_intrablock_additional_residuals
    adapter_state = pipe.adapter(adapter_input)
    for k, v in enumerate(adapter_state):
        adapter_state[k] = v * adapter_conditioning_scale
    if do_classifier_free_guidance:
        for k, v in enumerate(adapter_state):
            adapter_state[k] = torch.cat([v] * 2, dim=0)

    # 5. Prepare latent model input
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

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
        adapter_state,
    )
