# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HiDream-I1 model loader implementation for text-to-image generation.

Supports loading the HiDream-I1-Fast sparse diffusion transformer from
HiDream-ai/HiDream-I1-Fast using the diffusers HiDreamImagePipeline.
"""

from typing import Optional

import torch
from diffusers import HiDreamImagePipeline

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


class ModelVariant(StrEnum):
    """Available HiDream model variants."""

    FAST = "Fast"


class ModelLoader(ForgeModel):
    """HiDream-I1 model loader for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.FAST: ModelConfig(
            pretrained_model_name="HiDream-ai/HiDream-I1-Fast",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HiDream-I1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the HiDream image pipeline."""
        self._pipe = HiDreamImagePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self._pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the HiDream transformer model.

        Returns:
            torch.nn.Module: The HiDream sparse diffusion transformer.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, *, dtype_override=None, **kwargs):
        """Prepare transformer inputs for the HiDream model.

        Returns:
            dict: Input tensors for the transformer forward pass.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._pipe is None:
            self._load_pipeline(dtype)

        prompt = "A cat holding a sign that says hello world"
        height = 128
        width = 128
        num_inference_steps = 16

        # Encode prompts using the pipeline's text encoders
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            num_images_per_prompt=1,
        )

        # Prepare latent dimensions
        num_channels_latents = self._pipe.transformer.config.in_channels
        vae_scale_factor = self._pipe.vae_scale_factor
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor

        # Create random latents
        latents = torch.randn(1, num_channels_latents, latent_h, latent_w, dtype=dtype)

        # Prepare timestep
        timestep = torch.tensor([num_inference_steps // 2], dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds.to(dtype),
            "pooled_projections": pooled_prompt_embeds.to(dtype),
        }
