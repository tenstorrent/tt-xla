# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Blue Pencil FP16 XL (femboysLover/blue_pencil-fp16-XL) model loader implementation.

Blue Pencil FP16 XL is a text-to-image model based on SDXL, fine-tuned for
image generation with FP16 precision.

Available variants:
- BLUE_PENCIL_FP16_XL: femboysLover/blue_pencil-fp16-XL text-to-image generation
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


REPO_ID = "femboysLover/blue_pencil-fp16-XL"


class ModelVariant(StrEnum):
    """Available Blue Pencil FP16 XL model variants."""

    BLUE_PENCIL_FP16_XL = "blue_pencil_fp16_XL"


class ModelLoader(ForgeModel):
    """Blue Pencil FP16 XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.BLUE_PENCIL_FP16_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BLUE_PENCIL_FP16_XL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="blue_pencil_fp16_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Blue Pencil FP16 XL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Blue Pencil FP16 XL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Blue Pencil FP16 XL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
