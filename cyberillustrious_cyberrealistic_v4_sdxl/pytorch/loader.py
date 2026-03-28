# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CyberIllustrious CyberRealistic V4.0 SDXL model loader implementation.

CyberIllustrious CyberRealistic V4.0 is a Stable Diffusion XL merge model
for text-to-image generation, combining CyberIllustrious and CyberRealistic styles.

Available variants:
- CYBERILLUSTRIOUS_CYBERREALISTIC_V4_0: John6666/cyberillustrious-cyberrealistic-v40-sdxl text-to-image generation
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


REPO_ID = "John6666/cyberillustrious-cyberrealistic-v40-sdxl"


class ModelVariant(StrEnum):
    """Available CyberIllustrious CyberRealistic V4.0 SDXL model variants."""

    CYBERILLUSTRIOUS_CYBERREALISTIC_V4_0 = "CyberIllustrious_CyberRealistic_V4.0"


class ModelLoader(ForgeModel):
    """CyberIllustrious CyberRealistic V4.0 SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.CYBERILLUSTRIOUS_CYBERREALISTIC_V4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.CYBERILLUSTRIOUS_CYBERREALISTIC_V4_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CyberIllustrious_CyberRealistic_V4.0_SDXL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CyberIllustrious CyberRealistic V4.0 SDXL pipeline.

        Returns:
            DiffusionPipeline: The pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        self.pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A photorealistic portrait of a woman in cyberpunk city, detailed lighting, high quality"
        ] * batch_size
