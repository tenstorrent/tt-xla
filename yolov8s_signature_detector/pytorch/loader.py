# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOv8s Signature Detector (tech4humans/yolov8s-signature-detector) model loader implementation.

YOLOv8s-based object detection model fine-tuned to detect handwritten
signatures in document images.
"""
from typing import Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from torchvision import transforms
from datasets import load_dataset

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available YOLOv8s Signature Detector model variants."""

    YOLOV8S = "yolov8s"


class ModelLoader(ForgeModel):
    """YOLOv8s Signature Detector model loader implementation."""

    _VARIANTS = {
        ModelVariant.YOLOV8S: ModelConfig(
            pretrained_model_name="yolov8s.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YOLOV8S

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLOv8s Signature Detector",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download("tech4humans/yolov8s-signature-detector", filename)
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
