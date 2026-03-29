# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Helper functions for loading MLX-quantized FLUX.1-schnell models.
"""

import torch
from diffusers import FluxPipeline, FluxTransformer2DModel


def load_flux_mlx_pipe(repo_id: str, base_model: str):
    """Load a FLUX pipeline with an MLX-quantized transformer.

    Args:
        repo_id: HuggingFace repository ID containing the MLX model weights.
        base_model: HuggingFace repository ID of the base FLUX model for pipeline components.

    Returns:
        FluxPipeline: Loaded pipeline with MLX-quantized transformer.
    """
    transformer = FluxTransformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
    )

    pipe = FluxPipeline.from_pretrained(
        base_model,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )

    pipe.to("cpu")

    for module in [pipe.transformer, pipe.text_encoder, pipe.text_encoder_2, pipe.vae]:
        if module is not None:
            module.eval()
            for param in module.parameters():
                if param.requires_grad:
                    param.requires_grad = False

    return pipe


def flux_schnell_preprocessing(
    pipe,
    prompt,
    height=128,
    width=128,
    max_sequence_length=256,
    num_images_per_prompt=1,
    batch_size=1,
    dtype=None,
):
    """Preprocess inputs for the FLUX.1-schnell transformer model.

    Args:
        pipe: FLUX pipeline instance.
        prompt: Text prompt for generation.
        height: Output image height in pixels (default: 128).
        width: Output image width in pixels (default: 128).
        max_sequence_length: Maximum sequence length for T5 encoder (default: 256).
        num_images_per_prompt: Number of images per prompt (default: 1).
        batch_size: Batch size (default: 1).
        dtype: Torch dtype for inputs (default: bfloat16).

    Returns:
        dict: Input tensors for the FLUX transformer model.
    """
    if dtype is None:
        dtype = torch.bfloat16

    num_channels_latents = pipe.transformer.config.in_channels // 4

    # Text encoding for CLIP
    text_inputs_clip = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids_clip = text_inputs_clip.input_ids
    pooled_prompt_embeds = pipe.text_encoder(
        text_input_ids_clip, output_hidden_states=False
    ).pooler_output
    pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=dtype)
    pooled_prompt_embeds = pooled_prompt_embeds.repeat(
        batch_size, num_images_per_prompt
    )
    pooled_prompt_embeds = pooled_prompt_embeds.view(
        batch_size * num_images_per_prompt, -1
    )

    # Text encoding for T5
    text_inputs_t5 = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids_t5 = text_inputs_t5.input_ids
    prompt_embeds = pipe.text_encoder_2(text_input_ids_t5, output_hidden_states=False)[
        0
    ]
    prompt_embeds = prompt_embeds.to(dtype=dtype)
    _, seq_len_t5, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(batch_size, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(
        batch_size * num_images_per_prompt, seq_len_t5, -1
    )

    # Create text IDs
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(dtype=dtype)

    # Create latents
    height_latent = 2 * (int(height) // (pipe.vae_scale_factor * 2))
    width_latent = 2 * (int(width) // (pipe.vae_scale_factor * 2))

    shape = (
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height_latent,
        width_latent,
    )

    latents = torch.randn(shape, dtype=dtype)
    latents = latents.view(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height_latent // 2,
        2,
        width_latent // 2,
        2,
    )
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(
        batch_size * num_images_per_prompt,
        (height_latent // 2) * (width_latent // 2),
        num_channels_latents * 4,
    )

    # Prepare latent image IDs
    latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
    )
    latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

    return {
        "hidden_states": latents,
        "timestep": torch.tensor([1.0], dtype=dtype),
        "guidance": None,
        "pooled_projections": pooled_prompt_embeds,
        "encoder_hidden_states": prompt_embeds,
        "txt_ids": text_ids,
        "img_ids": latent_image_ids,
        "joint_attention_kwargs": {},
    }
