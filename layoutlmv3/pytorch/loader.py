# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LayoutLMv3 model loader implementation
"""
import torch
from transformers import LayoutLMv3Model, LayoutLMv3Processor
from typing import Optional
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
from datasets import load_dataset


class ModelVariant(StrEnum):
    """Available LayoutLMv3 model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """LayoutLMv3 model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="microsoft/layoutlmv3-base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Sample document words and their bounding boxes (normalized 0-1000)
    words = ["Hello", "world", "this", "is", "a", "sample", "document"]
    boxes = [
        [0, 0, 100, 50],
        [110, 0, 200, 50],
        [210, 0, 280, 50],
        [290, 0, 320, 50],
        [330, 0, 350, 50],
        [360, 0, 460, 50],
        [470, 0, 580, 50],
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LayoutLMv3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        processor_kwargs = {"apply_ocr": False}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self.processor = LayoutLMv3Processor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LayoutLMv3Model.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Create a simple white document image
        image = Image.new("RGB", (224, 224), color=(255, 255, 255))

        # Process with manually provided words and bounding boxes
        inputs = self.processor(
            image,
            self.words,
            boxes=self.boxes,
            return_tensors="pt",
        )

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
