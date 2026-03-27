# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for Stable Diffusion 3.5 FP8 model loading and processing.
"""

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download

REPO_ID = "Comfy-Org/stable-diffusion-3.5-fp8"


def load_pipe(filename, dtype=torch.float32):
    """Load Stable Diffusion 3.5 FP8 pipeline from a single-file safetensors checkpoint.

    Args:
        filename: Safetensors filename within the Comfy-Org/stable-diffusion-3.5-fp8 repo.
        dtype: Torch dtype for the pipeline.

    Returns:
        StableDiffusion3Pipeline: Loaded pipeline with components set to eval mode.
    """
    checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    pipe = StableDiffusion3Pipeline.from_single_file(
        checkpoint_path,
        torch_dtype=dtype,
    )
    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.transformer,
        pipe.text_encoder_2,
        pipe.vae,
    ]
    # text_encoder_3 (T5) may not be present in all single-file checkpoints
    if pipe.text_encoder_3 is not None:
        modules.append(pipe.text_encoder_3)

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def calculate_shift(
    image_seq_len,
    base_image_seq_len,
    max_image_seq_len,
    base_shift,
    max_shift,
):
    """Calculate dynamic shifting parameter for the scheduler."""
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    mu = image_seq_len * m + b
    return mu


def stable_diffusion_preprocessing_v35(
    pipe,
    prompt,
    device="cpu",
    negative_prompt=None,
    guidance_scale=7.0,
    num_inference_steps=1,
    num_images_per_prompt=1,
    clip_skip=None,
    max_sequence_length=256,
    do_classifier_free_guidance=True,
    mu=None,
):
    """Preprocess inputs for Stable Diffusion 3.5 FP8 model.

    Args:
        pipe: Stable Diffusion 3.5 pipeline.
        prompt: Text prompt for generation.
        device: Device to run on.
        negative_prompt: Negative prompt (optional).
        guidance_scale: Guidance scale.
        num_inference_steps: Number of inference steps.
        num_images_per_prompt: Number of images per prompt.
        clip_skip: CLIP skip layers (optional).
        max_sequence_length: Maximum sequence length.
        do_classifier_free_guidance: Whether to use classifier-free guidance.
        mu: Dynamic shifting parameter (optional).

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds)
    """
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor

    pipe.check_inputs(
        prompt,
        None,  # prompt_2
        None,  # prompt_3
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

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

    num_channels_latents = pipe.transformer.config.in_channels
    shape = (
        num_images_per_prompt,
        num_channels_latents,
        int(height) // pipe.vae_scale_factor,
        int(width) // pipe.vae_scale_factor,
    )
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)

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
        num_inference_steps=1,
        device=device,
        sigmas=None,
        **scheduler_kwargs,
    )

    latent_model_input = (
        torch.cat([latents] * 2) if do_classifier_free_guidance else latents
    )
    timestep = timesteps[0].expand(latent_model_input.shape[0])

    return latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds
