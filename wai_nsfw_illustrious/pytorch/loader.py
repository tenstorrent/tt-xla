# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
WAI NSFW Illustrious v140 (Ine007/waiNSFWIllustrious_v140) model loader implementation.

WAI NSFW Illustrious is a Stable Diffusion XL model finetuned from
Illustrious-XL-v1.0 for anime/illustration style text-to-image generation.

Available variants:
- WAI_NSFW_ILLUSTRIOUS_V140: Ine007/waiNSFWIllustrious_v140 text-to-image generation
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


REPO_ID = "Ine007/waiNSFWIllustrious_v140"


class ModelVariant(StrEnum):
    """Available WAI NSFW Illustrious model variants."""

    WAI_NSFW_ILLUSTRIOUS_V140 = "WAI_NSFW_Illustrious_v140"


class ModelLoader(ForgeModel):
    """WAI NSFW Illustrious model loader implementation."""

    _VARIANTS = {
        ModelVariant.WAI_NSFW_ILLUSTRIOUS_V140: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAI_NSFW_ILLUSTRIOUS_V140

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAI_NSFW_Illustrious_v140",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the WAI NSFW Illustrious pipeline.

        Returns:
            StableDiffusionXLPipeline: The WAI NSFW Illustrious pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the WAI NSFW Illustrious model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
