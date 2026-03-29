# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HuggingPanda Docling Layout model loader implementation for document layout detection.
"""
import torch
from PIL import Image
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor
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


class ModelVariant(StrEnum):
    """Available HuggingPanda Docling Layout model variants."""

    DOCLING_LAYOUT = "docling_layout"


class ModelLoader(ForgeModel):
    """HuggingPanda Docling Layout model loader for document layout detection tasks."""

    _VARIANTS = {
        ModelVariant.DOCLING_LAYOUT: ModelConfig(
            pretrained_model_name="HuggingPanda/docling-layout",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOCLING_LAYOUT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="HuggingPanda_Docling_Layout",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = RTDetrImageProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = RTDetrForObjectDetection.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image = Image.new("RGB", (640, 640), color=(128, 128, 128))
        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
