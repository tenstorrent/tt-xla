# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL-Turbo (stabilityai/sdxl-turbo) model loader implementation.

SDXL-Turbo is a fast text-to-image model based on SDXL 1.0, distilled using
Adversarial Diffusion Distillation (ADD) for single-step image generation.

Available variants:
- SDXL_TURBO: stabilityai/sdxl-turbo text-to-image generation
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


REPO_ID = "stabilityai/sdxl-turbo"


class ModelVariant(StrEnum):
    """Available SDXL-Turbo model variants."""

    SDXL_TURBO = "SDXL_Turbo"


class ModelLoader(ForgeModel):
    """SDXL-Turbo model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_TURBO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.SDXL_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL_Turbo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL-Turbo pipeline.

        Returns:
            AutoPipelineForText2Image: The SDXL-Turbo pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the SDXL-Turbo model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
