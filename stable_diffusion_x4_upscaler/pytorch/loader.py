# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion x4 Upscaler model loader implementation.

Loads the stabilityai/stable-diffusion-x4-upscaler pipeline, a text-guided
latent upscaling diffusion model that produces 4x upscaled images conditioned
on a low-resolution input image and a text prompt.

Available variants:
- BASE: stabilityai/stable-diffusion-x4-upscaler
"""

from typing import Optional

import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

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
    """Available Stable Diffusion x4 Upscaler model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion x4 Upscaler model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="stabilityai/stable-diffusion-x4-upscaler",
        ),
    }
    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion x4 Upscaler",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Diffusion x4 Upscaler pipeline.

        Returns:
            StableDiffusionUpscalePipeline: The pre-trained upscaling pipeline.
        """
        dtype = dtype_override or torch.bfloat16
        pipe = StableDiffusionUpscalePipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
            **kwargs,
        )
        return pipe

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the upscaler model.

        Returns:
            dict: Dictionary with 'prompt' (list of strings) and 'image'
                  (low-resolution PIL Image).
        """
        # Create a small low-resolution test image (128x128)
        low_res_image = Image.new("RGB", (128, 128), color=(128, 128, 128))

        prompt = ["a high quality, detailed photograph"] * batch_size

        return {"prompt": prompt, "image": low_res_image}
