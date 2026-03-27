# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RealVisXL V5.0 model loader implementation.

RealVisXL V5.0 is a photorealistic text-to-image model fine-tuned from
Stable Diffusion XL (SDXL). It uses the StableDiffusionXLPipeline from diffusers.

Available variants:
- REALVISXL_V5_0: SG161222/RealVisXL_V5.0 photorealistic text-to-image generation
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


REPO_ID = "SG161222/RealVisXL_V5.0"


class ModelVariant(StrEnum):
    """Available RealVisXL model variants."""

    REALVISXL_V5_0 = "RealVisXL_V5.0"


class ModelLoader(ForgeModel):
    """RealVisXL V5.0 model loader implementation."""

    _VARIANTS = {
        ModelVariant.REALVISXL_V5_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.REALVISXL_V5_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RealVisXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RealVisXL V5.0 pipeline.

        Returns:
            StableDiffusionXLPipeline: The RealVisXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the RealVisXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A photograph of a mountain landscape at golden hour, highly detailed, photorealistic"
        ] * batch_size
