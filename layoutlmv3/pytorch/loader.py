# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LayoutLMv3 model loader implementation
"""
import torch
import numpy as np
from transformers import LayoutLMv3Model, LayoutLMv3TokenizerFast
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
        self.tokenizer = None

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

    def _load_tokenizer(self):
        self.tokenizer = LayoutLMv3TokenizerFast.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LayoutLMv3Model.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Tokenize words with bounding boxes
        inputs = self.tokenizer(self.words, boxes=self.boxes, return_tensors="pt")

        # Create a simple white document image and convert to pixel_values
        image = Image.new("RGB", (224, 224), color=(255, 255, 255))
        pixel_values = (
            torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        inputs["pixel_values"] = pixel_values

        # Handle batch size
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
