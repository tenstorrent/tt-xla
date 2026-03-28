# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DocumentFigureClassifier model loader implementation for document figure classification.
"""
import torch
from PIL import Image
from transformers import EfficientNetForImageClassification
from torchvision import transforms
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
    """Available DocumentFigureClassifier model variants."""

    V2_0 = "v2.0"


class ModelLoader(ForgeModel):
    """DocumentFigureClassifier model loader for document figure classification tasks."""

    _VARIANTS = {
        ModelVariant.V2_0: ModelConfig(
            pretrained_model_name="docling-project/DocumentFigureClassifier-v2.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DocumentFigureClassifier",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = EfficientNetForImageClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.47853944, 0.4732864, 0.47434163],
                ),
            ]
        )

        image = Image.new("RGB", (640, 480), color=(128, 128, 128))
        pixel_values = preprocess(image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
