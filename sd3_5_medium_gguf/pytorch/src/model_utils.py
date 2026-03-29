# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading GGUF-quantized Stable Diffusion 3.5 models.
"""

from typing import Optional, Tuple

import torch
from diffusers import DiffusionPipeline, GGUFQuantizationConfig
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)
from huggingface_hub import hf_hub_download


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


def load_gguf_pipe(repo_id: str, gguf_filename: str):
    """Load a Stable Diffusion 3.5 pipeline from a GGUF checkpoint.

    Args:
        repo_id: HuggingFace repository ID.
        gguf_filename: Filename of the GGUF checkpoint within the repo.

    Returns:
        DiffusionPipeline: Loaded pipeline with components set to eval mode.
    """
    model_path = hf_hub_download(repo_id=repo_id, filename=gguf_filename)

    quantization_config = GGUFQuantizationConfig(compute_dtype=torch.float32)

    pipe = DiffusionPipeline.from_single_file(
        model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.float32,
    )

    pipe.to("cpu")

    for module in [pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
        if module is not None:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    if hasattr(pipe, "text_encoder_3") and pipe.text_encoder_3 is not None:
        pipe.text_encoder_3.eval()
        for param in pipe.text_encoder_3.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


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
    """Preprocess inputs for Stable Diffusion v3.5 model.

    Args:
        pipe: Stable Diffusion v3.5 pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        negative_prompt: Negative prompt (optional)
        guidance_scale: Guidance scale (default: 7.0)
        num_inference_steps: Number of inference steps (default: 1)
        num_images_per_prompt: Number of images per prompt (default: 1)
        clip_skip: CLIP skip layers (optional)
        max_sequence_length: Maximum sequence length (default: 256)
        do_classifier_free_guidance: Whether to use classifier-free guidance (default: True)
        mu: Dynamic shifting parameter (optional)

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
