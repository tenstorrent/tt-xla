# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
True Pencil XL (John6666/true-pencil-xl-v100-sdxl) model loader implementation.

True Pencil XL is a fine-tuned SDXL model specialized for generating
anime/illustration-style images with a pencil-drawing aesthetic.

Available variants:
- TRUE_PENCIL_XL: John6666/true-pencil-xl-v100-sdxl text-to-image generation
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


REPO_ID = "John6666/true-pencil-xl-v100-sdxl"


class ModelVariant(StrEnum):
    """Available True Pencil XL model variants."""

    TRUE_PENCIL_XL = "True_Pencil_XL"


class ModelLoader(ForgeModel):
    """True Pencil XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.TRUE_PENCIL_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.TRUE_PENCIL_XL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="True_Pencil_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the True Pencil XL pipeline.

        Returns:
            StableDiffusionXLPipeline: The True Pencil XL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the True Pencil XL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
