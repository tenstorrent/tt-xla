# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAM2 (Segment Anything Model 2) loader implementation
"""

import torch
from typing import Optional
from PIL import Image
from loguru import logger
from transformers import Sam2Model, Sam2Processor

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
    """Available SAM2 model variants."""

    HIERA_TINY = "Hiera_Tiny"
    HIERA_LARGE = "Hiera_Large"


class ModelLoader(ForgeModel):
    """SAM2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.HIERA_TINY: ModelConfig(
            pretrained_model_name="facebook/sam2.1-hiera-tiny",
        ),
        ModelVariant.HIERA_LARGE: ModelConfig(
            pretrained_model_name="facebook/sam2.1-hiera-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HIERA_TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SAM2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        framework_model = Sam2Model.from_pretrained(model_name, **kwargs).to("cpu")

        self.processor = Sam2Processor.from_pretrained(model_name, **kwargs)

        if dtype_override is not None:
            framework_model = framework_model.to(dtype_override)

        return framework_model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = Sam2Processor.from_pretrained(model_name)

        try:
            dataset = load_dataset("huggingface/cats-image")["test"]
            raw_image = dataset[0]["image"].convert("RGB")
        except Exception as e:
            logger.warning(
                f"Failed to load image from dataset. Using random fallback tensor. Reason: {e}"
            )
            raw_image = Image.fromarray(
                (torch.rand(3, 1024, 1024) * 255).byte().permute(1, 2, 0).numpy()
            )

        input_points = [[[450, 600]]]

        inputs = self.processor(
            raw_image, input_points=input_points, return_tensors="pt"
        ).to("cpu")

        pixel_values = inputs["pixel_values"]
        input_points_tensor = inputs["input_points"]

        pixel_values = pixel_values.repeat_interleave(batch_size, dim=0)
        input_points_tensor = input_points_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)
            input_points_tensor = input_points_tensor.to(dtype_override)

        return pixel_values, input_points_tensor
