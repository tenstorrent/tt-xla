# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLO26 (rujutashashikanjoshi/yolo26-testH-vehicle-detection-4931_full-100m) model loader implementation.

YOLO26 Medium model fine-tuned for single-class vehicle (car) detection,
trained on a custom dataset for 100 epochs.
"""
from typing import Optional

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from torchvision import transforms
from ultralytics import YOLO

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available YOLO26 model variants."""

    YOLO26_MEDIUM = "medium"


class ModelLoader(ForgeModel):
    """YOLO26 model loader implementation."""

    _VARIANTS = {
        ModelVariant.YOLO26_MEDIUM: ModelConfig(
            pretrained_model_name="rujutashashikanjoshi/yolo26-testH-vehicle-detection-4931_full-100m",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YOLO26_MEDIUM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLO26",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_name = self._variant_config.pretrained_model_name
        weights_path = hf_hub_download(repo_id=pretrained_name, filename="best.pt")
        yolo_model = YOLO(weights_path)
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
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )
        batch_tensor = preprocess(image).unsqueeze(0)
        batch_tensor = batch_tensor.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            batch_tensor = batch_tensor.to(dtype_override)

        return batch_tensor
