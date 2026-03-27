# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for ControlNet Depth SD3.5 model loading and processing.
"""

import torch
from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from PIL import Image


def load_controlnet_depth_sd3_pipe(controlnet_model_name, base_model_name):
    """Load ControlNet Depth SD3.5 pipeline.

    Args:
        controlnet_model_name: ControlNet model name on HuggingFace
        base_model_name: Base SD3.5 model name on HuggingFace

    Returns:
        StableDiffusion3ControlNetPipeline: Loaded pipeline with components set to eval mode
    """
    controlnet = SD3ControlNetModel.from_pretrained(
        controlnet_model_name, torch_dtype=torch.float32
    )
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        base_model_name, controlnet=controlnet, torch_dtype=torch.float32
    )

    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.transformer,
        pipe.text_encoder_2,
        pipe.text_encoder_3,
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
    return Image.new("RGB", (width, height), color=(128, 128, 128))


def calculate_shift(
    image_seq_len,
    base_image_seq_len,
    max_image_seq_len,
    base_shift,
    max_shift,
):
    """Calculate dynamic shifting parameter for the scheduler.

    Args:
        image_seq_len: Current image sequence length
        base_image_seq_len: Base image sequence length
        max_image_seq_len: Maximum image sequence length
        base_shift: Base shift value
        max_shift: Maximum shift value

    Returns:
        float: Calculated shift parameter
    """
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = image_seq_len * m + b
    return mu


def controlnet_depth_sd3_preprocessing(
    pipe,
    prompt,
    control_image,
    device="cpu",
    negative_prompt=None,
    guidance_scale=4.5,
    num_inference_steps=1,
    num_images_per_prompt=1,
    clip_skip=None,
    max_sequence_length=77,
    controlnet_conditioning_scale=1.0,
    do_classifier_free_guidance=True,
    mu=None,
):
    """Preprocess inputs for ControlNet Depth SD3.5 model.

    Args:
        pipe: StableDiffusion3ControlNetPipeline
        prompt: Text prompt for generation
        control_image: Depth conditioning image
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 4.5)
        num_inference_steps: Number of inference steps (default: 1)
        num_images_per_prompt: Number of images per prompt (default: 1)
        clip_skip: CLIP skip layers (optional)
        max_sequence_length: Maximum sequence length (default: 77)
        controlnet_conditioning_scale: ControlNet conditioning scale (default: 1.0)
        do_classifier_free_guidance: Whether to use classifier-free guidance (default: True)
        mu: Dynamic shifting parameter (optional)

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds,
                controlnet_block_samples)
    """
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor

    # 1. Encode the prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        device=device,
        clip_skip=clip_skip,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pooled_prompt_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
        )

    # 2. Prepare timesteps with dynamic shifting
    scheduler_kwargs = {}
    if pipe.scheduler.config.get("use_dynamic_shifting", None) and mu is None:
        image_seq_len = (height // pipe.transformer.config.patch_size) * (
            width // pipe.transformer.config.patch_size
        )
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.base_image_seq_len,
            pipe.scheduler.config.max_image_seq_len,
            pipe.scheduler.config.base_shift,
            pipe.scheduler.config.max_shift,
        )
        scheduler_kwargs["mu"] = mu
    elif mu is not None:
        scheduler_kwargs["mu"] = mu

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=None,
        **scheduler_kwargs,
    )

    # 3. Prepare latent variables
    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        num_images_per_prompt,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    torch.manual_seed(42)
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)

    # 4. Prepare latent model input
    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    # 5. Prepare control image
    control_image = pipe.prepare_image(
        image=control_image,
        width=width,
        height=height,
        batch_size=num_images_per_prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        dtype=pipe.controlnet.dtype,
        do_classifier_free_guidance=do_classifier_free_guidance,
    )

    # 6. Run controlnet to get block samples
    controlnet_block_samples = pipe.controlnet(
        hidden_states=latent_model_input,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        controlnet_cond=control_image,
        conditioning_scale=controlnet_conditioning_scale,
        return_dict=False,
    )[0]

    return (
        latent_model_input,
        timestep,
        prompt_embeds,
        pooled_prompt_embeds,
        controlnet_block_samples,
    )
