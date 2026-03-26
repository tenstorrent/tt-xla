# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViTMatte model loader implementation for image matting tasks.
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

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


class ModelVariant(StrEnum):
    """Available ViTMatte model variants."""

    SMALL_COMPOSITION_1K = "Small_Composition_1k"


class ModelLoader(ForgeModel):
    """ViTMatte model loader implementation for image matting tasks."""

    _VARIANTS = {
        ModelVariant.SMALL_COMPOSITION_1K: ModelConfig(
            pretrained_model_name="hustvl/vitmatte-small-composition-1k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMALL_COMPOSITION_1K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ViTMatte",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = VitMatteImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        model_kwargs |= kwargs

        model = VitMatteForImageMatting.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        # Load a test image from HuggingFace dataset
        dataset = load_dataset("huggingface/cats-image", split="test")
        image = dataset[0]["image"]

        # Create a synthetic trimap: all pixels marked as unknown (128)
        # This gives the model a non-trivial matting task
        trimap = Image.fromarray(
            np.full((image.height, image.width), 128, dtype=np.uint8)
        )

        inputs = self.processor(images=image, trimaps=trimap, return_tensors="pt")

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        return inputs
