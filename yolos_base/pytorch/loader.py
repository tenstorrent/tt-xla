# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOS-Base model loader implementation for object detection.
"""

import torch
from transformers import (
    YolosImageProcessor,
    YolosForObjectDetection,
)
from typing import Optional

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
from datasets import load_dataset


class ModelLoader(ForgeModel):
    """YOLOS-Base model loader implementation for object detection tasks."""

    _VARIANTS = {}
    DEFAULT_VARIANT = None

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_variant = "hustvl/yolos-base"
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="YOLOS Base",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = YolosForObjectDetection.from_pretrained(self.model_variant, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self.processor = YolosImageProcessor.from_pretrained(self.model_variant)

        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        inputs = self.processor(images=image, return_tensors="pt")
        batch_tensor = inputs["pixel_values"]

        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
