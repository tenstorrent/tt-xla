# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoMAE model loader implementation for video classification.
"""

from typing import Optional

import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

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
    """Available VideoMAE model variants."""

    BASE_KINETICS = "Base_Kinetics"


class ModelLoader(ForgeModel):
    """VideoMAE model loader for video classification."""

    _VARIANTS = {
        ModelVariant.BASE_KINETICS: ModelConfig(
            pretrained_model_name="MCG-NJU/videomae-base-finetuned-kinetics",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_KINETICS

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoMAE model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoMAE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoMAE model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = VideoMAEForVideoClassification.from_pretrained(
            str(model_name), **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoMAE.

        Generates a synthetic video of 16 frames at 224x224 resolution.
        """
        if self.processor is None:
            model_name = self._variant_config.pretrained_model_name
            self.processor = VideoMAEImageProcessor.from_pretrained(model_name)

        # Create a synthetic video (16 frames of 224x224 RGB)
        video = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)
        ]

        inputs = self.processor(video, return_tensors="pt")

        if dtype_override:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        if batch_size > 1:
            inputs = {
                k: v.repeat(batch_size, *([1] * (v.dim() - 1)))
                if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()
            }

        return dict(inputs)
