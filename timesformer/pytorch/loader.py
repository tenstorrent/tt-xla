# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TimeSformer model loader implementation
"""

from typing import Optional
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification

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


@dataclass
class TimeSformerConfig(ModelConfig):
    """Configuration specific to TimeSformer models"""

    num_frames: int = 8


class ModelVariant(StrEnum):
    """Available TimeSformer model variants."""

    BASE_FINETUNED_K600 = "Base_Finetuned_K600"


class ModelLoader(ForgeModel):
    """TimeSformer model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE_FINETUNED_K600: TimeSformerConfig(
            pretrained_model_name="facebook/timesformer-base-finetuned-k600",
            num_frames=8,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_FINETUNED_K600

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="TimeSformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = TimesformerForVideoClassification.from_pretrained(
            pretrained_model_name, **kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        pretrained_model_name = self._variant_config.pretrained_model_name
        num_frames = self._variant_config.num_frames

        processor = AutoImageProcessor.from_pretrained(pretrained_model_name)

        # Create synthetic video input: list of frames as numpy arrays
        video = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(num_frames)
        ]

        inputs = processor(video, return_tensors="pt")
        pixel_values = inputs["pixel_values"]

        if batch_size > 1:
            pixel_values = pixel_values.expand(batch_size, -1, -1, -1, -1)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return pixel_values
