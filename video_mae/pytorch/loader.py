# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoMAE model loader implementation for video feature extraction.
"""

from typing import Optional

import numpy as np
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor

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
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available VideoMAE model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """VideoMAE model loader for video self-supervised pre-training."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="MCG-NJU/videomae-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

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

    def _load_processor(self):
        model_name = self._variant_config.pretrained_model_name
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        return self.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the VideoMAE model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = VideoMAEForPreTraining.from_pretrained(str(model_name), **kwargs)
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoMAE."""
        if self.processor is None:
            self._load_processor()

        # Create synthetic video: 16 frames of 224x224 RGB
        video = list(np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8))

        inputs = self.processor(video, return_tensors="pt")

        if batch_size > 1:
            inputs = {
                k: v.repeat_interleave(batch_size, dim=0) for k, v in inputs.items()
            }

        if dtype_override:
            inputs = {
                k: cast_input_to_type(v, dtype_override) for k, v in inputs.items()
            }

        return dict(inputs)
