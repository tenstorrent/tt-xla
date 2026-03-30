# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YOLOE (jameslahm/yoloe-v8l-seg) model loader implementation.

YOLOE is a real-time, high-accuracy object detection and instance
segmentation model built on the YOLOv8 architecture with zero-shot
capabilities via text, visual, or prompt-free modes.
"""
from typing import Optional

from datasets import load_dataset
from torchvision import transforms
from ultralytics import YOLO

from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available YOLOE model variants."""

    YOLOE_V8L_SEG = "v8l-seg"


class ModelLoader(ForgeModel):
    """YOLOE model loader implementation."""

    _VARIANTS = {
        ModelVariant.YOLOE_V8L_SEG: ModelConfig(
            pretrained_model_name="jameslahm/yoloe-v8l-seg",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YOLOE_V8L_SEG

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="YOLOE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_name = self._variant_config.pretrained_model_name
        yolo_model = YOLO(pretrained_name)
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
