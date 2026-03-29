# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNet50 Facial Emotion Recognition model loader implementation
"""

import torch
import numpy as np
from transformers import AutoModelForImageClassification
from torchvision import transforms
from typing import Optional

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    ModelConfig,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available ResNet50 Facial Emotion Recognition model variants."""

    RESNET50_FER = "resnet50_fer"


class ModelLoader(ForgeModel):
    """ResNet50 Facial Emotion Recognition model loader implementation."""

    _VARIANTS = {
        ModelVariant.RESNET50_FER: ModelConfig(
            pretrained_model_name="KhaldiAbderrhmane/resnet50-facial-emotion-recognition",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RESNET50_FER

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ResNet50_Facial_Emotion_Recognition",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()

        self.model = model

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, *, dtype_override=None, batch_size=1, **kwargs):
        preprocess = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Create a synthetic face image (224x224 RGB)
        face_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pixel_values = preprocess(face_image).unsqueeze(0)

        if batch_size > 1:
            pixel_values = pixel_values.expand(batch_size, -1, -1, -1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
