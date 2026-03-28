# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RealVisXL V3.0 (SG161222/RealVisXL_V3.0) model loader implementation.

RealVisXL V3.0 is a photorealism-focused text-to-image model built on
Stable Diffusion XL (SDXL).

Available variants:
- REALVISXL_V3_0: SG161222/RealVisXL_V3.0 text-to-image generation
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


REPO_ID = "SG161222/RealVisXL_V3.0"


class ModelVariant(StrEnum):
    """Available RealVisXL V3.0 model variants."""

    REALVISXL_V3_0 = "RealVisXL_V3.0"


class ModelLoader(ForgeModel):
    """RealVisXL V3.0 model loader implementation."""

    _VARIANTS = {
        ModelVariant.REALVISXL_V3_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.REALVISXL_V3_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RealVisXL_V3.0",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the RealVisXL V3.0 pipeline.

        Returns:
            StableDiffusionXLPipeline: The RealVisXL V3.0 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the RealVisXL V3.0 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
