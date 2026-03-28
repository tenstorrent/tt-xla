# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nova Anime XL (John6666/nova-anime-xl-il-v80-sdxl) model loader implementation.

Nova Anime XL is an SDXL-based model fine-tuned for anime and illustration
style image generation, merged from Illustrious-XL-v2.0 and noobai-XL-1.1.

Available variants:
- NOVA_ANIME_XL: John6666/nova-anime-xl-il-v80-sdxl text-to-image generation
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


REPO_ID = "John6666/nova-anime-xl-il-v80-sdxl"


class ModelVariant(StrEnum):
    """Available Nova Anime XL model variants."""

    NOVA_ANIME_XL = "Nova_Anime_XL"


class ModelLoader(ForgeModel):
    """Nova Anime XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.NOVA_ANIME_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NOVA_ANIME_XL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Nova_Anime_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Nova Anime XL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Nova Anime XL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Nova Anime XL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A beautiful anime girl in a fantasy landscape with colorful flowers"
        ] * batch_size
