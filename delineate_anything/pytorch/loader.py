# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DelineateAnything model loader implementation.

Resolution-agnostic deep learning model for detecting and delineating
agricultural field boundaries in satellite imagery, built on Ultralytics.
"""
from typing import Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchvision import transforms
from ultralytics import YOLO

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
    Framework,
)


class ModelVariant(StrEnum):
    """Available DelineateAnything model variants."""

    DELINEATE_ANYTHING_S = "DelineateAnything-S"
    DELINEATE_ANYTHING = "DelineateAnything"


class ModelLoader(ForgeModel):
    """DelineateAnything model loader implementation."""

    _VARIANTS = {
        ModelVariant.DELINEATE_ANYTHING_S: ModelConfig(
            pretrained_model_name="DelineateAnything-S.pt",
        ),
        ModelVariant.DELINEATE_ANYTHING: ModelConfig(
            pretrained_model_name="DelineateAnything.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DELINEATE_ANYTHING

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DelineateAnything",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download("MykolaL/DelineateAnything", filename)
        yolo_model = YOLO(model_path)
        model = yolo_model.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize((640, 640)),
                transforms.ToTensor(),
            ]
        )
        batch_tensor = preprocess(image).unsqueeze(0)
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
