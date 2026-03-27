# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Anzhc's YOLOs model loader implementation
"""
import torch
from torchvision import transforms
from datasets import load_dataset
from typing import Optional

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
from ...tools.utils import yolo_postprocess


class ModelVariant(StrEnum):
    """Available Anzhc's YOLOs model variants."""

    FACE_SEG_V4_Y11N = "Face Seg v4 y11n"
    DRONE_DET_V03_Y11N = "Drone Det v03 y11n"


class ModelLoader(ForgeModel):
    """Anzhc's YOLOs model loader implementation."""

    _VARIANTS = {
        ModelVariant.FACE_SEG_V4_Y11N: ModelConfig(
            pretrained_model_name="Anzhc Face seg 640 v4 y11n.pt",
        ),
        ModelVariant.DRONE_DET_V03_Y11N: ModelConfig(
            pretrained_model_name="Anzhcs Drones v03 1024 y11n.pt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FACE_SEG_V4_Y11N

    _INPUT_SIZES = {
        ModelVariant.FACE_SEG_V4_Y11N: 640,
        ModelVariant.DRONE_DET_V03_Y11N: 1024,
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Anzhcs_YOLOs",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from ultralytics import YOLO
        from huggingface_hub import hf_hub_download

        filename = self._variant_config.pretrained_model_name
        model_path = hf_hub_download(
            repo_id="Anzhc/Anzhcs_YOLOs",
            filename=filename,
        )

        yolo = YOLO(model_path)
        model = yolo.model.float()
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        input_size = self._INPUT_SIZES[self._variant]

        dataset = load_dataset("huggingface/cats-image", split="test[:1]")
        image = dataset[0]["image"]
        preprocess = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
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
