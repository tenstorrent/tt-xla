# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FLUX FP8 model loader implementation for text-to-image generation.

Loads FP8-quantized FLUX.1 transformer weights from Kijai/flux-fp8.
Avoids accessing gated black-forest-labs/FLUX.1-* repos by using a local
transformer config and generating synthetic inputs.
"""
import os
import json
import tempfile

import torch
from diffusers.models import FluxTransformer2DModel
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)

# HuggingFace URLs for FP8 checkpoint files (Kijai repo, not gated)
_FP8_DEV_URL = (
    "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-dev-fp8-e4m3fn.safetensors"
)
_FP8_SCHNELL_URL = "https://huggingface.co/Kijai/flux-fp8/blob/main/flux1-schnell-fp8-e4m3fn.safetensors"

# Transformer config for FLUX models. Architecture is the same for Dev and
# Schnell; the only difference is guidance_embeds (True for Dev).
_TRANSFORMER_CONFIG = {
    "_class_name": "FluxTransformer2DModel",
    "_diffusers_version": "0.37.1",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "in_channels": 64,
    "joint_attention_dim": 4096,
    "num_attention_heads": 24,
    "num_layers": 19,
    "num_single_layers": 38,
    "patch_size": 1,
    "pooled_projection_dim": 768,
}


class ModelVariant(StrEnum):
    """Available FLUX FP8 model variants."""

    DEV = "Dev"
    SCHNELL = "Schnell"


class ModelLoader(ForgeModel):
    """FLUX FP8 model loader for text-to-image generation using FP8-quantized weights."""

    _VARIANTS = {
        ModelVariant.DEV: ModelConfig(
            pretrained_model_name="Kijai/flux-fp8",
        ),
        ModelVariant.SCHNELL: ModelConfig(
            pretrained_model_name="Kijai/flux-fp8",
        ),
    }

    _FP8_URLS = {
        ModelVariant.DEV: _FP8_DEV_URL,
        ModelVariant.SCHNELL: _FP8_SCHNELL_URL,
    }

    DEFAULT_VARIANT = ModelVariant.DEV

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None
        self.guidance_scale = 3.5 if self._variant == ModelVariant.DEV else 0.0

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FLUX-FP8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _make_local_config_dir(self):
        """Create a temporary directory with the transformer config.json."""
        config = dict(_TRANSFORMER_CONFIG)
        config["guidance_embeds"] = self._variant == ModelVariant.DEV

        config_dir = tempfile.mkdtemp()
        transformer_dir = os.path.join(config_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)
        with open(os.path.join(transformer_dir, "config.json"), "w") as f:
            json.dump(config, f)
        return config_dir

    def _load_transformer(self, dtype_override=None):
        """Load the FP8-quantized FLUX transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        config_dir = self._make_local_config_dir()

        self._transformer = FluxTransformer2DModel.from_single_file(
            self._FP8_URLS[self._variant],
            config=config_dir,
            subfolder="transformer",
            torch_dtype=dtype,
        )

        return self._transformer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the FP8-quantized FLUX transformer model."""
        if self._transformer is None:
            self._load_transformer(dtype_override=dtype_override)

        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype_override)

        return self._transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Generate sample inputs for the FLUX FP8 transformer model."""
        if self._transformer is None:
            self._load_transformer(dtype_override=dtype_override)

        max_sequence_length = 256
        do_classifier_free_guidance = self.guidance_scale > 1.0
        height = 128
        width = 128
        num_images_per_prompt = 1
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        num_channels_latents = self._transformer.config.in_channels // 4
        vae_scale_factor = 8

        # Pooled CLIP text embeddings
        pooled_projection_dim = self._transformer.config.pooled_projection_dim
        pooled_prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt, pooled_projection_dim, dtype=dtype
        )

        # T5 text encoder hidden states
        joint_attention_dim = self._transformer.config.joint_attention_dim
        prompt_embeds = torch.randn(
            batch_size * num_images_per_prompt,
            max_sequence_length,
            joint_attention_dim,
            dtype=dtype,
        )

        # Text IDs
        text_ids = torch.zeros(max_sequence_length, 3, dtype=dtype)

        # Latents (packed format)
        height_latent = 2 * (int(height) // (vae_scale_factor * 2))
        width_latent = 2 * (int(width) // (vae_scale_factor * 2))

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

        # Latent image IDs
        latent_image_ids = torch.zeros(height_latent // 2, width_latent // 2, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(height_latent // 2)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(width_latent // 2)[None, :]
        )
        latent_image_ids = latent_image_ids.reshape(-1, 3).to(dtype=dtype)

        # Guidance
        if do_classifier_free_guidance:
            guidance = torch.full([batch_size], self.guidance_scale, dtype=dtype)
        else:
            guidance = None

        inputs = {
            "hidden_states": latents,
            "timestep": torch.tensor([1.0], dtype=dtype),
            "guidance": guidance,
            "pooled_projections": pooled_prompt_embeds,
            "encoder_hidden_states": prompt_embeds,
            "txt_ids": text_ids,
            "img_ids": latent_image_ids,
            "joint_attention_kwargs": {},
        }

        return inputs
