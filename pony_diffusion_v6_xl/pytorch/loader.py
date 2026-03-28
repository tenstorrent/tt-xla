# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pony Diffusion V6 XL model loader implementation.

Pony Diffusion V6 XL is a Stable Diffusion XL fine-tune for text-to-image
generation, trained on ~2.6M images with detailed captions.

Available variants:
- PONY_DIFFUSION_V6_XL: LyliaEngine/Pony_Diffusion_V6_XL text-to-image generation
"""

from typing import Optional

import torch
from diffusers import DiffusionPipeline

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


REPO_ID = "LyliaEngine/Pony_Diffusion_V6_XL"


class ModelVariant(StrEnum):
    """Available Pony Diffusion V6 XL model variants."""

    PONY_DIFFUSION_V6_XL = "Pony_Diffusion_V6_XL"


class ModelLoader(ForgeModel):
    """Pony Diffusion V6 XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.PONY_DIFFUSION_V6_XL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PONY_DIFFUSION_V6_XL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Pony_Diffusion_V6_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Pony Diffusion V6 XL pipeline.

        Returns:
            DiffusionPipeline: The Pony Diffusion V6 XL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Pony Diffusion V6 XL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "score_9, score_8_up, score_7_up, a beautiful landscape with mountains and a lake, rating_safe"
        ] * batch_size
