# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bubble Hentai Illustrious (John6666/bubble-hentai-illustrious-v10-sdxl) model loader implementation.

Bubble Hentai Illustrious is a text-to-image generation model based on Stable Diffusion XL,
fine-tuned for anime-style illustration generation.

Available variants:
- BUBBLE_HENTAI_ILLUSTRIOUS_V10: John6666/bubble-hentai-illustrious-v10-sdxl text-to-image generation
"""

import torch
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
from .src.model_utils import load_pipe, stable_diffusion_preprocessing_xl


class ModelVariant(StrEnum):
    """Available Bubble Hentai Illustrious model variants."""

    BUBBLE_HENTAI_ILLUSTRIOUS_V10 = "bubble-hentai-illustrious-v10"


class ModelLoader(ForgeModel):
    """Bubble Hentai Illustrious model loader implementation."""

    _VARIANTS = {
        ModelVariant.BUBBLE_HENTAI_ILLUSTRIOUS_V10: ModelConfig(
            pretrained_model_name="John6666/bubble-hentai-illustrious-v10-sdxl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BUBBLE_HENTAI_ILLUSTRIOUS_V10

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Bubble Hentai Illustrious",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Bubble Hentai Illustrious pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            DiffusionPipeline: The Bubble Hentai Illustrious pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Bubble Hentai Illustrious model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            List : Input tensors that can be fed to the model:
                - latent_model_input (torch.Tensor): Latent input for the UNet
                - timestep (torch.Tensor): Timestep tensor
                - prompt_embeds (torch.Tensor): Encoded prompt embeddings
                - added_cond_kwargs (dict): Additional conditioning inputs
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
