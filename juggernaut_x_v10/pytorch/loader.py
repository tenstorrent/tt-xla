# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Juggernaut X v10 (RunDiffusion/Juggernaut-X-v10) model loader implementation.

Juggernaut X v10 is a text-to-image model based on SDXL, fine-tuned from
stabilityai/stable-diffusion-xl-base-1.0 using GPT-4 Vision captioning.

Available variants:
- JUGGERNAUT_X_V10: RunDiffusion/Juggernaut-X-v10 text-to-image generation
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


REPO_ID = "RunDiffusion/Juggernaut-X-v10"


class ModelVariant(StrEnum):
    """Available Juggernaut X v10 model variants."""

    JUGGERNAUT_X_V10 = "Juggernaut_X_v10"


class ModelLoader(ForgeModel):
    """Juggernaut X v10 model loader implementation."""

    _VARIANTS = {
        ModelVariant.JUGGERNAUT_X_V10: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.JUGGERNAUT_X_V10

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Juggernaut_X_v10",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Juggernaut X v10 pipeline.

        Returns:
            StableDiffusionXLPipeline: The Juggernaut X v10 pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Juggernaut X v10 model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
