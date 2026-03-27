#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pixel Art Style LoRA model loader implementation.

Loads the Z-Image-Turbo base pipeline and applies Pixel Art Style LoRA weights
from tarn59/pixel_art_style_lora_z_image_turbo for stylized text-to-image generation.

Available variants:
- BASE: Default pixel art style LoRA on Z-Image-Turbo
"""

from typing import Optional

import torch
from diffusers import AutoPipelineForText2Image  # type: ignore[import]

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

BASE_MODEL = "Tongyi-MAI/Z-Image-Turbo"
LORA_REPO = "tarn59/pixel_art_style_lora_z_image_turbo"


class ModelVariant(StrEnum):
    """Available Pixel Art Style LoRA variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Pixel Art Style LoRA model loader."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="PIXEL_ART_STYLE_LORA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Z-Image-Turbo pipeline with Pixel Art Style LoRA weights applied.

        Returns:
            Pipeline with LoRA weights loaded.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        pipe.load_lora_weights(LORA_REPO)
        return pipe

    def load_inputs(self, batch_size=1, **kwargs):
        """Load and return sample text prompts for pixel art generation.

        Returns:
            list: A list of sample text prompts.
        """
        prompt = [
            "Pixel art style. A small cottage in a forest clearing with warm light glowing from the windows.",
        ] * batch_size
        return prompt
