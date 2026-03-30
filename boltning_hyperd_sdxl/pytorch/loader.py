# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Boltning HyperD SDXL model loader implementation.

Boltning HyperD SDXL is a speed-optimized Stable Diffusion XL checkpoint for
fast text-to-image generation.

Available variants:
- BOLTNING_HYPERD_SDXL: GraydientPlatformAPI/boltning-hyperd-sdxl text-to-image generation
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


class ModelVariant(StrEnum):
    """Available Boltning HyperD SDXL model variants."""

    BOLTNING_HYPERD_SDXL = "boltning-hyperd-sdxl"


class ModelLoader(ForgeModel):
    """Boltning HyperD SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.BOLTNING_HYPERD_SDXL: ModelConfig(
            pretrained_model_name="GraydientPlatformAPI/boltning-hyperd-sdxl",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BOLTNING_HYPERD_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Boltning_HyperD_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Boltning HyperD SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The Boltning HyperD SDXL pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Boltning HyperD SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic photo of a lighthouse on a cliff during a storm, dramatic lighting, high detail"
        ] * batch_size
