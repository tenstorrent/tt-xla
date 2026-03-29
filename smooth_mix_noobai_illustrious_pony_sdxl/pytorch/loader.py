# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Smooth Mix NoobAI Illustrious Pony SDXL
(John6666/smooth-mix-noobai-illustrious-pony-illustrious2-noobai-v2-sdxl)
model loader implementation.

This is a Stable Diffusion XL merge model combining NoobAI-XL and
Illustrious-XL for text-to-image generation.

Available variants:
- SMOOTH_MIX_NOOBAI_ILLUSTRIOUS_PONY_SDXL: Text-to-image generation
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


REPO_ID = "John6666/smooth-mix-noobai-illustrious-pony-illustrious2-noobai-v2-sdxl"


class ModelVariant(StrEnum):
    """Available Smooth Mix NoobAI Illustrious Pony SDXL model variants."""

    SMOOTH_MIX_NOOBAI_ILLUSTRIOUS_PONY_SDXL = "Smooth_Mix_NoobAI_Illustrious_Pony_SDXL"


class ModelLoader(ForgeModel):
    """Smooth Mix NoobAI Illustrious Pony SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.SMOOTH_MIX_NOOBAI_ILLUSTRIOUS_PONY_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SMOOTH_MIX_NOOBAI_ILLUSTRIOUS_PONY_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Smooth_Mix_NoobAI_Illustrious_Pony_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Smooth Mix NoobAI Illustrious Pony SDXL pipeline.

        Returns:
            AutoPipelineForText2Image: The pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
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
