# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Docling document layout analysis model loader implementation for object detection.
"""
import torch
from PIL import Image
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
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
    """Available Docling layout model variants for document layout analysis."""

    HERON = "Heron"
    EGRET_LARGE = "Egret_Large"


class ModelLoader(ForgeModel):
    """Docling layout model loader for document layout analysis tasks."""

    _VARIANTS = {
        ModelVariant.HERON: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-heron",
        ),
        ModelVariant.EGRET_LARGE: ModelConfig(
            pretrained_model_name="docling-project/docling-layout-egret-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HERON

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Docling_Layout",
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

    def _load_model_class(self):
        """Return the appropriate model class for the current variant."""
        if self._variant == ModelVariant.EGRET_LARGE:
            from transformers import DFineForObjectDetection

            return DFineForObjectDetection
        return RTDetrV2ForObjectDetection

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.processor is None:
            self._load_processor()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model_class = self._load_model_class()
        model = model_class.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        inputs = self.processor(images=image, return_tensors="pt")

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
                if dtype_override is not None:
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
