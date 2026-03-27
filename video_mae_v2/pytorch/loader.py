# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VideoMAEv2 model loader implementation for video feature extraction.
"""

from typing import Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, VideoMAEImageProcessor

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


class ModelVariant(StrEnum):
    """Available VideoMAEv2 model variants."""

    HUGE = "Huge"


class ModelLoader(ForgeModel):
    """VideoMAEv2 model loader for video feature extraction."""

    _VARIANTS = {
        ModelVariant.HUGE: ModelConfig(
            pretrained_model_name="OpenGVLab/VideoMAEv2-Huge",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HUGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize VideoMAEv2 model loader."""
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="VideoMAEv2",
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
        """Load and return the VideoMAEv2 model instance."""
        model_name = self._variant_config.pretrained_model_name
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            model_name, config=config, trust_remote_code=True, **kwargs
        )
        model.eval()

        if dtype_override:
            model = model.to(dtype_override)

        if self.processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for VideoMAEv2."""
        if self.processor is None:
            self._load_processor()

        # Create synthetic video: 16 frames of 224x224 RGB
        video = list(np.random.rand(16, 3, 224, 224))
        inputs = self.processor(video, return_tensors="pt")

        # VideoMAEv2 expects (B, C, T, H, W) format
        inputs["pixel_values"] = inputs["pixel_values"].permute(0, 2, 1, 3, 4)

        if dtype_override:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return dict(inputs)
