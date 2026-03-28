# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Best Comic Panel Detection (mosesb/best-comic-panel-detection) model loader implementation.

YOLOv12x-based comic panel detection model fine-tuned to detect
individual panels on comic book pages.
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
    Framework,
    StrEnum,
)
from ...tools.utils import yolo_postprocess


class ModelVariant(StrEnum):
    """Available Best Comic Panel Detection model variants."""

    YOLOv12X = "yolo12x"


class ModelLoader(ForgeModel):
    """Best Comic Panel Detection model loader implementation."""

    _VARIANTS = {
        ModelVariant.YOLOv12X: ModelConfig(
            pretrained_model_name="best.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.YOLOv12X

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Best Comic Panel Detection",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download("mosesb/best-comic-panel-detection", filename)
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

    def post_process(self, co_out):
        return yolo_postprocess(co_out)
