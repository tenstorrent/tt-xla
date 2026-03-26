# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiVOLO V2 model loader implementation
"""

import torch
import numpy as np
from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
)
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
    """Available MiVOLO V2 model variants."""

    V2 = "v2"


class ModelLoader(ForgeModel):
    """MiVOLO V2 model loader implementation."""

    _VARIANTS = {
        ModelVariant.V2: ModelConfig(
            pretrained_model_name="iitolstykh/mivolo_v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MiVOLO_V2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None
        self._image_processor = None

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

    def _get_image_processor(self):
        if self._image_processor is None:
            model_name = self._variant_config.pretrained_model_name
            self._image_processor = AutoImageProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
        return self._image_processor

    def load_inputs(self, *, dtype_override=None, batch_size=1, **kwargs):
        processor = self._get_image_processor()

        # Create a synthetic face crop (384x384 RGB)
        face_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        body_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)

        faces_crops = [face_image] * batch_size
        bodies_crops = [body_image] * batch_size

        faces_input = processor(images=faces_crops)["pixel_values"]
        body_input = processor(images=bodies_crops)["pixel_values"]

        if dtype_override is not None:
            faces_input = faces_input.to(dtype_override)
            body_input = body_input.to(dtype_override)

        return {"faces_input": faces_input, "body_input": body_input}
