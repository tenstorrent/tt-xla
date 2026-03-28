# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hassaku XL Illustrious (John6666/hassaku-xl-illustrious-v21-sdxl) model loader implementation.

Hassaku XL Illustrious is an anime/illustration-focused SDXL fine-tune based on
OnomaAIResearch/Illustrious-xl-early-release-v0 for text-to-image generation.

Available variants:
- HASSAKU_XL_ILLUSTRIOUS_V21: John6666/hassaku-xl-illustrious-v21-sdxl text-to-image generation
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
from ...stable_diffusion_xl.pytorch.src.model_utils import (
    load_pipe,
    stable_diffusion_preprocessing_xl,
)


class ModelVariant(StrEnum):
    """Available Hassaku XL Illustrious model variants."""

    HASSAKU_XL_ILLUSTRIOUS_V21 = "hassaku-xl-illustrious-v21-sdxl"


class ModelLoader(ForgeModel):
    """Hassaku XL Illustrious model loader implementation."""

    _VARIANTS = {
        ModelVariant.HASSAKU_XL_ILLUSTRIOUS_V21: ModelConfig(
            pretrained_model_name="John6666/hassaku-xl-illustrious-v21-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HASSAKU_XL_ILLUSTRIOUS_V21

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Hassaku XL Illustrious",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hassaku XL Illustrious pipeline.

        Returns:
            DiffusionPipeline: The Hassaku XL Illustrious pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Hassaku XL Illustrious model.

        Returns:
            List: Input tensors for the UNet model.
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
