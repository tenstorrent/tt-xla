# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DucHaiten-AIart-SDXL_v3 (openart-custom/DucHaiten-AIart-SDXL_v3) model loader implementation.

DucHaiten-AIart-SDXL_v3 is a fine-tuned Stable Diffusion XL model for artistic
text-to-image generation.

Available variants:
- DUCHAITEN_AIART_SDXL_V3: openart-custom/DucHaiten-AIart-SDXL_v3 text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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


REPO_ID = "openart-custom/DucHaiten-AIart-SDXL_v3"


class ModelVariant(StrEnum):
    """Available DucHaiten-AIart-SDXL model variants."""

    DUCHAITEN_AIART_SDXL_V3 = "DucHaiten_AIart_SDXL_v3"


class ModelLoader(ForgeModel):
    """DucHaiten-AIart-SDXL_v3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.DUCHAITEN_AIART_SDXL_V3: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DUCHAITEN_AIART_SDXL_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DucHaiten_AIart_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DucHaiten-AIart-SDXL_v3 pipeline.

        Returns:
            StableDiffusionXLPipeline: The SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
