# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD-Turbo model loader implementation
"""

import torch
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from diffusers import StableDiffusionPipeline


class ModelVariant(StrEnum):
    """Available SD-Turbo model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """SD-Turbo model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stabilityai/sd-turbo",
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="SD-Turbo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SD-Turbo pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            StableDiffusionPipeline: The pre-trained SD-Turbo pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, torch_dtype=dtype, **kwargs
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for SD-Turbo.

        Args:
            dtype_override: This parameter is ignored for this model.
            batch_size: Optional batch size for the prompts.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        ] * batch_size
        return prompt
