# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FastHunyuan video diffusion model loading and processing.
"""

import torch
from diffusers import HunyuanVideoPipeline


def load_pipe(model_name):
    """Load HunyuanVideo pipeline.

    Args:
        model_name: HuggingFace model identifier

    Returns:
        HunyuanVideoPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = HunyuanVideoPipeline.from_pretrained(model_name, torch_dtype=torch.float32)

    modules = [
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.transformer,
        pipe.vae,
    ]

    pipe.to("cpu")

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def hunyuan_video_preprocessing(
    pipe,
    prompt,
    device="cpu",
    num_inference_steps=1,
    height=128,
    width=128,
    num_frames=9,
):
    """Preprocess inputs for HunyuanVideo model.

    Args:
        pipe: HunyuanVideo pipeline
        prompt: Text prompt for generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 1)
        height: Video frame height (default: 128)
        width: Video frame width (default: 128)
        num_frames: Number of video frames (default: 9)

    Returns:
        tuple: (latent_model_input, timestep, prompt_embeds, prompt_attention_mask,
                guidance)
    """
    # Encode prompt
    prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_videos_per_prompt=1,
        prompt_template=None,
    )

    # Create latent noise
    num_channels_latents = pipe.transformer.config.in_channels
    vae_scale_factor_spatial = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    vae_scale_factor_temporal = pipe.vae.config.temporal_compression_ratio

    latent_height = height // vae_scale_factor_spatial
    latent_width = width // vae_scale_factor_spatial
    latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1

    shape = (
        1,
        num_channels_latents,
        latent_frames,
        latent_height,
        latent_width,
    )
    latents = torch.randn(shape, device=device, dtype=prompt_embeds.dtype)

    # Setup scheduler and get timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    timestep = timesteps[0].expand(latents.shape[0])

    # Guidance scale
    guidance = torch.tensor([6.0], device=device, dtype=prompt_embeds.dtype)

    return latents, timestep, prompt_embeds, prompt_attention_mask, guidance
