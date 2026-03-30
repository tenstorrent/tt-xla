# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WAI NSFW Illustrious SDXL v140 (dhead/wai-nsfw-illustrious-sdxl-v140-sdxl) model loader implementation.

WAI NSFW Illustrious SDXL is a Stable Diffusion XL model fine-tuned for
photorealistic text-to-image generation.

Available variants:
- WAI_NSFW_ILLUSTRIOUS_SDXL_V140: dhead/wai-nsfw-illustrious-sdxl-v140-sdxl text-to-image generation
"""

from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline

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


REPO_ID = "dhead/wai-nsfw-illustrious-sdxl-v140-sdxl"


class ModelVariant(StrEnum):
    """Available WAI NSFW Illustrious SDXL model variants."""

    WAI_NSFW_ILLUSTRIOUS_SDXL_V140 = "WAI_NSFW_Illustrious_SDXL_v140"


class ModelLoader(ForgeModel):
    """WAI NSFW Illustrious SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAI_NSFW_ILLUSTRIOUS_SDXL_V140: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAI_NSFW_ILLUSTRIOUS_SDXL_V140

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAI_NSFW_Illustrious_SDXL_v140",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WAI NSFW Illustrious SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The WAI NSFW Illustrious SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the WAI NSFW Illustrious SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
