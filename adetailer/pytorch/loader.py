# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Adetailer (Bingsu/adetailer) model loader implementation.

YOLOv8-based face/hand/person detection models commonly used
as an after-detailer in Stable Diffusion pipelines.
"""
import torch
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
    """Available Adetailer model variants."""

    FACE_YOLOV8N = "face_yolov8n"
    FACE_YOLOV8S = "face_yolov8s"
    HAND_YOLOV8N = "hand_yolov8n"
    PERSON_YOLOV8N_SEG = "person_yolov8n-seg"


class ModelLoader(ForgeModel):
    """Adetailer model loader implementation."""

    _VARIANTS = {
        ModelVariant.FACE_YOLOV8N: ModelConfig(
            pretrained_model_name="face_yolov8n.pt",
        ),
        ModelVariant.FACE_YOLOV8S: ModelConfig(
            pretrained_model_name="face_yolov8s.pt",
        ),
        ModelVariant.HAND_YOLOV8N: ModelConfig(
            pretrained_model_name="hand_yolov8n.pt",
        ),
        ModelVariant.PERSON_YOLOV8N_SEG: ModelConfig(
            pretrained_model_name="person_yolov8n-seg.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FACE_YOLOV8N

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Adetailer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download("Bingsu/adetailer", filename)
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
