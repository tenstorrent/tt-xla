# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Animagine XL 3.1 (votepurchase/animagine-xl-3.1) model loader implementation.

Animagine XL 3.1 is an anime-focused text-to-image model based on Stable Diffusion XL,
fine-tuned for high-quality anime-style image generation.

Available variants:
- ANIMAGINE_XL_3_1: votepurchase/animagine-xl-3.1 text-to-image generation
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


REPO_ID = "votepurchase/animagine-xl-3.1"


class ModelVariant(StrEnum):
    """Available Animagine XL 3.1 model variants."""

    ANIMAGINE_XL_3_1 = "Animagine_XL_3_1"


class ModelLoader(ForgeModel):
    """Animagine XL 3.1 model loader implementation."""

    _VARIANTS = {
        ModelVariant.ANIMAGINE_XL_3_1: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.ANIMAGINE_XL_3_1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Animagine_XL_3_1",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Animagine XL 3.1 pipeline.

        Returns:
            StableDiffusionXLPipeline: The Animagine XL 3.1 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Animagine XL 3.1 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "1girl, green hair, sweater, looking at viewer, upper body, beanie, outdoors, night, turtleneck, masterpiece, best quality, very aesthetic"
        ] * batch_size
