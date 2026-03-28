# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCM-LoRA SDXL model loader implementation.

LCM-LoRA (Latent Consistency Model LoRA) is a distilled consistency adapter for
SDXL that reduces inference from 20-50 steps down to 2-8 steps while maintaining
image quality.

Available variants:
- LCM_LORA_SDXL: latent-consistency/lcm-lora-sdxl text-to-image generation
"""

from typing import Optional

import torch

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
from .src.model_utils import load_pipe, lcm_lora_sdxl_preprocessing


class ModelVariant(StrEnum):
    """Available LCM-LoRA SDXL model variants."""

    LCM_LORA_SDXL = "LCM_LoRA_SDXL"


class ModelLoader(ForgeModel):
    """LCM-LoRA SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.LCM_LORA_SDXL: ModelConfig(
            pretrained_model_name="latent-consistency/lcm-lora-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LCM_LORA_SDXL

    prompt = "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LCM-LoRA SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the LCM-LoRA SDXL pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            AutoPipelineForText2Image: The LCM-LoRA SDXL pipeline instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.pipeline = load_pipe(pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the LCM-LoRA SDXL model.

        Args:
            dtype_override: Optional torch.dtype to override input dtype.

        Returns:
            list: Input tensors for the UNet:
                - latent_model_input (torch.Tensor)
                - timestep (torch.Tensor)
                - prompt_embeds (torch.Tensor)
                - added_cond_kwargs (dict)
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
        ) = lcm_lora_sdxl_preprocessing(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
