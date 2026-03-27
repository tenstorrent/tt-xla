# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ViTPose model loader implementation for pose estimation
"""

import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation
from typing import Optional
from PIL import Image
import numpy as np

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
    """Available ViTPose model variants."""

    BASE_SIMPLE = "Base_Simple"


class ModelLoader(ForgeModel):
    """ViTPose model loader implementation for pose estimation tasks."""

    _VARIANTS = {
        ModelVariant.BASE_SIMPLE: ModelConfig(
            pretrained_model_name="usyd-community/vitpose-base-simple",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_SIMPLE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ViTPose",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_KEYPOINT_DET,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.processor is None:
            self._load_processor()

        model = VitPoseForPoseEstimation.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.processor is None:
            self._load_processor()

        # Create a synthetic image with a person-like bounding box
        image = Image.fromarray(
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        )
        boxes = [[100, 100, 200, 300]]  # COCO format: [x, y, width, height]

        inputs = self.processor(image, boxes=[boxes], return_tensors="pt")

        if dtype_override is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype_override)

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Add dataset_index (0 = COCO dataset decoder)
        inputs["dataset_index"] = torch.tensor([0]).repeat_interleave(batch_size, dim=0)

        return inputs
