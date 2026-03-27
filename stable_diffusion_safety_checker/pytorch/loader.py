# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion Safety Checker model loader implementation
"""

from typing import Optional
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available Stable Diffusion Safety Checker model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Stable Diffusion Safety Checker model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="CompVis/stable-diffusion-safety-checker",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="StableDiffusionSafetyChecker",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = StableDiffusionSafetyChecker.from_pretrained(
            pretrained_model_name, **kwargs
        )
        model.eval()

        self.model = model

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]

        if self._processor is None:
            model_name = self._variant_config.pretrained_model_name
            self._processor = CLIPImageProcessor.from_pretrained(model_name)

        pixel_values = self._processor(images=image, return_tensors="pt")[
            "pixel_values"
        ]

        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values, pixel_values
