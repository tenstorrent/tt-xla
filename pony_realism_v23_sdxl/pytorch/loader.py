# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pony Realism V2.3 SDXL (John6666/pony-realism-v23-sdxl) model loader implementation.

Pony Realism V2.3 is a photorealism-focused text-to-image model built on
Stable Diffusion XL (SDXL).

Available variants:
- PONY_REALISM_V2_3_SDXL: John6666/pony-realism-v23-sdxl text-to-image generation
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


REPO_ID = "John6666/pony-realism-v23-sdxl"


class ModelVariant(StrEnum):
    """Available Pony Realism V2.3 SDXL model variants."""

    PONY_REALISM_V2_3_SDXL = "pony-realism-v23-sdxl"


class ModelLoader(ForgeModel):
    """Pony Realism V2.3 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.PONY_REALISM_V2_3_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.PONY_REALISM_V2_3_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="pony-realism-v23-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Pony Realism V2.3 SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Pony Realism V2.3 SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            use_safetensors=True,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Pony Realism V2.3 SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
