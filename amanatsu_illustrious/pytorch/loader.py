# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Amanatsu Illustrious v1.1 SDXL (John6666/amanatsu-illustrious-v11-sdxl) model loader implementation.

Amanatsu Illustrious is an anime-focused Stable Diffusion XL fine-tune
optimized for detailed anime illustration generation.

Available variants:
- AMANATSU_ILLUSTRIOUS_V11: John6666/amanatsu-illustrious-v11-sdxl text-to-image generation
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


REPO_ID = "John6666/amanatsu-illustrious-v11-sdxl"


class ModelVariant(StrEnum):
    """Available Amanatsu Illustrious model variants."""

    AMANATSU_ILLUSTRIOUS_V11 = "amanatsu-illustrious-v11"


class ModelLoader(ForgeModel):
    """Amanatsu Illustrious SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.AMANATSU_ILLUSTRIOUS_V11: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.AMANATSU_ILLUSTRIOUS_V11

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Amanatsu Illustrious",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Amanatsu Illustrious SDXL pipeline.

        Returns:
            AutoPipelineForText2Image: The Amanatsu Illustrious pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Amanatsu Illustrious model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
