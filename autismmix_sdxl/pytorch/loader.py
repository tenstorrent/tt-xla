# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AutismMix SDXL (John6666/autismmix-sdxl-autismmix-pony-sdxl) model loader implementation.

AutismMix SDXL is an anime/pony-style text-to-image model based on the
Stable Diffusion XL architecture.

Available variants:
- AUTISMMIX_PONY_SDXL: John6666/autismmix-sdxl-autismmix-pony-sdxl text-to-image generation
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


REPO_ID = "John6666/autismmix-sdxl-autismmix-pony-sdxl"


class ModelVariant(StrEnum):
    """Available AutismMix SDXL model variants."""

    AUTISMMIX_PONY_SDXL = "AutismMix_Pony_SDXL"


class ModelLoader(ForgeModel):
    """AutismMix SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.AUTISMMIX_PONY_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.AUTISMMIX_PONY_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AutismMix_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AutismMix SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The AutismMix SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the AutismMix SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
