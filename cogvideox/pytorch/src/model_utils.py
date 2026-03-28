# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for CogVideoX model loading and preprocessing.
"""

import torch
from diffusers import CogVideoXPipeline


def load_pipe(pretrained_model_name):
    """Load CogVideoX pipeline.

    Args:
        pretrained_model_name: HuggingFace model identifier

    Returns:
        CogVideoXPipeline: Loaded pipeline with components set to eval mode
    """
    pipe = CogVideoXPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32
    )

    modules = [
        pipe.text_encoder,
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


def cogvideox_preprocessing(
    pipe,
    prompt,
    device="cpu",
    num_inference_steps=1,
    num_videos_per_prompt=1,
    guidance_scale=6.0,
    num_frames=9,
):
    """Preprocess inputs for CogVideoX transformer model.

    Args:
        pipe: CogVideoX pipeline
        prompt: Text prompt for video generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 1)
        num_videos_per_prompt: Number of videos per prompt (default: 1)
        guidance_scale: Guidance scale (default: 6.0)
        num_frames: Number of video frames to generate (default: 9)

    Returns:
        tuple: (hidden_states, timestep, encoder_hidden_states, image_rotary_emb)
    """
    do_classifier_free_guidance = guidance_scale > 1.0

    # Encode prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=None,
        do_classifier_free_guidance=do_classifier_free_guidance,
        num_videos_per_prompt=num_videos_per_prompt,
        device=device,
        dtype=torch.float32,
    )

    # Prepare latents
    num_channels_latents = pipe.transformer.config.in_channels
    latent_height = pipe.transformer.config.sample_height * 2
    latent_width = pipe.transformer.config.sample_width * 2

    # CogVideoX uses temporal compression in the VAE
    # The temporal compression factor is typically 4
    latent_num_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

    shape = (
        num_videos_per_prompt,
        latent_num_frames,
        num_channels_latents,
        latent_height,
        latent_width,
    )
    latents = torch.randn(shape, device=device, dtype=torch.float32)

    # Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # Get the first timestep
    timestep = timesteps[0].expand(latents.shape[0])

    # Concatenate for classifier-free guidance
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        latent_model_input = torch.cat([latents] * 2)
        timestep = timesteps[0].expand(latent_model_input.shape[0])
    else:
        latent_model_input = latents

    # Prepare rotary embeddings
    image_rotary_emb = pipe.transformer.patch_embed.get_rotary_embedding(
        latent_model_input, prompt_embeds
    )

    return latent_model_input, timestep, prompt_embeds, image_rotary_emb
