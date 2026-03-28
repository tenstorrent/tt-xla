# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku Z-Image-Turbo model loader implementation.

Nunchaku Z-Image-Turbo is a 4-bit quantized version of Tongyi-MAI/Z-Image-Turbo,
using SVDQuant for efficient text-to-image generation.

Available variants:
- NUNCHAKU_Z_IMAGE_TURBO: nunchaku-ai/nunchaku-z-image-turbo text-to-image generation
"""

from typing import Optional

import torch
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from nunchaku.models.z_image import NunchakuZImageTransformer2DModel

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


REPO_ID = "nunchaku-ai/nunchaku-z-image-turbo"
BASE_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


class ModelVariant(StrEnum):
    """Available Nunchaku Z-Image-Turbo model variants."""

    NUNCHAKU_Z_IMAGE_TURBO = "Nunchaku_Z_Image_Turbo"


class ModelLoader(ForgeModel):
    """Nunchaku Z-Image-Turbo model loader implementation."""

    _VARIANTS = {
        ModelVariant.NUNCHAKU_Z_IMAGE_TURBO: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.NUNCHAKU_Z_IMAGE_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Nunchaku_Z_Image_Turbo",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Nunchaku Z-Image-Turbo pipeline.

        Returns:
            ZImagePipeline: The Nunchaku Z-Image-Turbo pipeline instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        transformer = NunchakuZImageTransformer2DModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            subfolder="svdq-int4-r32",
            torch_dtype=dtype,
        )

        self.pipeline = ZImagePipeline.from_pretrained(
            BASE_MODEL_ID,
            transformer=transformer,
            torch_dtype=dtype,
            **kwargs,
        )

        return self.pipeline

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample text prompts for the Nunchaku Z-Image-Turbo model.

        Returns:
            list: A list of sample text prompts.
        """
        return [
            "A cinematic shot of a baby raccoon wearing an intricate italian priest robe."
        ] * batch_size
