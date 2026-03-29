# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HS UltraHD CG Illustration Epic SDXL (John6666/hs-ultrahd-cg-ill-epic-sdxl) model loader implementation.

HS UltraHD CG Illustration Epic is an illustration and CG art focused text-to-image
model built on Stable Diffusion XL (SDXL), fine-tuned from Illustrious XL.

Available variants:
- HS_ULTRAHD_CG_ILL_EPIC_SDXL: John6666/hs-ultrahd-cg-ill-epic-sdxl text-to-image generation
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


REPO_ID = "John6666/hs-ultrahd-cg-ill-epic-sdxl"


class ModelVariant(StrEnum):
    """Available HS UltraHD CG Illustration Epic SDXL model variants."""

    HS_ULTRAHD_CG_ILL_EPIC_SDXL = "hs-ultrahd-cg-ill-epic-sdxl"


class ModelLoader(ForgeModel):
    """HS UltraHD CG Illustration Epic SDXL model loader implementation."""

    _VARIANTS = {
        ModelVariant.HS_ULTRAHD_CG_ILL_EPIC_SDXL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.HS_ULTRAHD_CG_ILL_EPIC_SDXL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="hs-ultrahd-cg-ill-epic-sdxl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the HS UltraHD CG Illustration Epic SDXL pipeline.

        Returns:
            StableDiffusionXLPipeline: The HS UltraHD CG Illustration Epic SDXL pipeline instance.
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
        """Load and return sample text prompts for the HS UltraHD CG Illustration Epic SDXL model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
