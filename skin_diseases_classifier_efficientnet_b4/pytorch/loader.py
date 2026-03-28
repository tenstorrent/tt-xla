# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Skin Diseases Classifier EfficientNetB4 model loader implementation for image classification.

This model is a Keras-based EfficientNet-B4 fine-tuned for skin disease classification.
Source: https://huggingface.co/Vamsi232/Skin_Diseases_Classifier_EfficientNetB4_best
"""
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from datasets import load_dataset
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
    """Available Skin Diseases Classifier model variants."""

    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """Skin Diseases Classifier EfficientNetB4 model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="Vamsi232/Skin_Diseases_Classifier_EfficientNetB4_best",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Skin Diseases Classifier EfficientNetB4",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        import keras

        pretrained_model_name = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(
            pretrained_model_name,
            "Skin_Diseases_Classifier_EfficientNetB4_best.keras",
        )
        model = keras.models.load_model(model_path)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image")["test"]
        image = dataset[0]["image"].convert("RGB")

        # EfficientNet-B4 expects 380x380 images
        image = image.resize((380, 380))

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(image).astype(np.float32) / 255.0

        # Convert HWC to CHW format and add batch dimension
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        if batch_size > 1:
            img_tensor = img_tensor.repeat(batch_size, 1, 1, 1)

        if dtype_override is not None:
            img_tensor = img_tensor.to(dtype_override)

        return img_tensor
