# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DreamShaper XL Turbo model loader implementation.

DreamShaper XL Turbo is a fine-tuned SDXL model distilled for fast text-to-image
generation with very few inference steps.

Available variants:
- DREAMSHAPER_XL_V2_TURBO: Lykon/dreamshaper-xl-v2-turbo text-to-image generation
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image

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


class ModelVariant(StrEnum):
    """Available DreamShaper XL model variants."""

    DREAMSHAPER_XL_V2_TURBO = "DreamShaper_XL_v2_Turbo"


class ModelLoader(ForgeModel):
    """DreamShaper XL Turbo model loader implementation."""

    _VARIANTS = {
        ModelVariant.DREAMSHAPER_XL_V2_TURBO: ModelConfig(
            pretrained_model_name="Lykon/dreamshaper-xl-v2-turbo",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.DREAMSHAPER_XL_V2_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DreamShaper_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DreamShaper XL Turbo pipeline.

        Returns:
            AutoPipelineForText2Image: The DreamShaper XL Turbo pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the DreamShaper XL Turbo model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors"
        ] * batch_size
