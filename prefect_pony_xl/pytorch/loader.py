# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Prefect Pony XL v5.0 (John6666/prefect-pony-xl-v50-sdxl) model loader implementation.

Prefect Pony XL is a fine-tuned Stable Diffusion XL checkpoint optimized for
anime/pony-style image generation.

Available variants:
- PREFECT_PONY_XL_V5_0: John6666/prefect-pony-xl-v50-sdxl text-to-image generation
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


REPO_ID = "John6666/prefect-pony-xl-v50-sdxl"


class ModelVariant(StrEnum):
    """Available Prefect Pony XL model variants."""

    PREFECT_PONY_XL_V5_0 = "Prefect_Pony_XL_v5.0"


class ModelLoader(ForgeModel):
    """Prefect Pony XL model loader implementation."""

    _VARIANTS = {
        ModelVariant.PREFECT_PONY_XL_V5_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PREFECT_PONY_XL_V5_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Prefect_Pony_XL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Prefect Pony XL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Prefect Pony XL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Prefect Pony XL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
