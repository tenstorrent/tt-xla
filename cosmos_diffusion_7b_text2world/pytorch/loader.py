#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cosmos 1.0 Diffusion 7B Text2World model loader implementation.

NVIDIA Cosmos is a 7B-parameter Diffusion Transformer for text-to-video
generation (World Foundation Model). It generates physics-aware video from
text prompts using the Hugging Face Diffusers CosmosTextToWorldPipeline.

Available variants:
- COSMOS_7B_TEXT2WORLD: nvidia/Cosmos-1.0-Diffusion-7B-Text2World
"""

from typing import Optional

import torch
from diffusers import CosmosTextToWorldPipeline  # type: ignore[import]

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

REPO_ID = "nvidia/Cosmos-1.0-Diffusion-7B-Text2World"


class ModelVariant(StrEnum):
    """Available Cosmos Diffusion 7B Text2World model variants."""

    COSMOS_7B_TEXT2WORLD = "7B_Text2World"


class ModelLoader(ForgeModel):
    """Cosmos 1.0 Diffusion 7B Text2World model loader."""

    _VARIANTS = {
        ModelVariant.COSMOS_7B_TEXT2WORLD: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.COSMOS_7B_TEXT2WORLD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Cosmos_Diffusion_7B_Text2World",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16):
        """Load the CosmosTextToWorldPipeline."""
        self._pipe = CosmosTextToWorldPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Cosmos diffusion transformer.

        Returns:
            The diffusion transformer module from the pipeline.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        elif dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype=dtype_override)
        return self._pipe.transformer

    def load_inputs(self, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Prepare sample inputs for the Cosmos diffusion transformer.

        Returns:
            dict: Input tensors for the transformer model.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._pipe is None:
            self._load_pipeline(dtype)

        prompt = "A sleek humanoid robot stands in a vast warehouse"

        # Encode text prompt
        text_inputs = self._pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self._pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_embeds = self._pipe.text_encoder(text_inputs.input_ids)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # Create small latent noise input
        # Cosmos uses a 3D VAE: [batch, channels, temporal, height, width]
        num_channels = self._pipe.transformer.config.in_channels
        latents = torch.randn(1, num_channels, 2, 8, 8, dtype=dtype)

        timestep = torch.tensor([1.0], dtype=dtype)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
        }
