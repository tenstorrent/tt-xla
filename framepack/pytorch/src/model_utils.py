# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Helper functions for FramePack model loading and preprocessing.
"""

import torch
from diffusers import (
    HunyuanVideoFramepackPipeline,
    HunyuanVideoFramepackTransformer3DModel,
)
from transformers import SiglipImageProcessor, SiglipVisionModel


def load_pipe(pretrained_model_name):
    """Load FramePack pipeline.

    The FramePack pipeline requires assembling several components:
    - The FramePack transformer from the specified pretrained model
    - SigLIP image encoder for image conditioning
    - Base HunyuanVideo pipeline components (text encoders, VAE, scheduler)

    Args:
        pretrained_model_name: HuggingFace model identifier for the FramePack transformer

    Returns:
        HunyuanVideoFramepackPipeline: Loaded pipeline with components set to eval mode
    """
    transformer = HunyuanVideoFramepackTransformer3DModel.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float32
    )

    feature_extractor = SiglipImageProcessor.from_pretrained(
        "lllyasviel/flux_redux_bfl", subfolder="feature_extractor"
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        "lllyasviel/flux_redux_bfl",
        subfolder="image_encoder",
        torch_dtype=torch.float32,
    )

    pipe = HunyuanVideoFramepackPipeline.from_pretrained(
        "hunyuanvideo-community/HunyuanVideo",
        transformer=transformer,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=torch.float32,
    )

    pipe.to("cpu")

    modules = [
        pipe.text_encoder,
        pipe.text_encoder_2,
        pipe.transformer,
        pipe.vae,
        pipe.image_encoder,
    ]

    for module in modules:
        module.eval()
        for param in module.parameters():
            if param.requires_grad:
                param.requires_grad = False

    return pipe


def framepack_preprocessing(
    pipe,
    prompt,
    device="cpu",
    num_inference_steps=1,
    height=64,
    width=64,
    num_frames=5,
):
    """Preprocess inputs for FramePack transformer model.

    Args:
        pipe: HunyuanVideoFramepackPipeline
        prompt: Text prompt for video generation
        device: Device to run on (default: "cpu")
        num_inference_steps: Number of inference steps (default: 1)
        height: Height of the output video frames (default: 64)
        width: Width of the output video frames (default: 64)
        num_frames: Number of video frames (default: 5)

    Returns:
        tuple: (hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                pooled_projections, image_embeds, indices_latents)
    """
    # Encode text prompt
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_videos_per_prompt=1,
    )

    # Create a dummy input image for image conditioning (expected in [-1, 1] range)
    dummy_image = torch.rand(1, 3, height, width) * 2 - 1
    image_embeds = pipe.encode_image(dummy_image, device=device)

    # Prepare latent noise
    num_channels_latents = pipe.transformer.config.in_channels
    latent_height = height // pipe.vae_scale_factor_spatial
    latent_width = width // pipe.vae_scale_factor_spatial

    shape = (1, num_channels_latents, num_frames, latent_height, latent_width)
    hidden_states = torch.randn(shape, device=device, dtype=torch.float32)

    # Prepare timestep
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    timestep = timesteps[0].expand(hidden_states.shape[0])

    # Prepare frame indices
    indices_latents = torch.arange(0, num_frames).unsqueeze(0)

    return (
        hidden_states,
        timestep,
        prompt_embeds,
        prompt_attention_mask,
        pooled_prompt_embeds,
        image_embeds,
        indices_latents,
    )
