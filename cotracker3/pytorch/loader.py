# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
CoTracker3 model loader implementation for video point tracking.
"""

from typing import Optional

import torch

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
    """Available CoTracker3 model variants."""

    OFFLINE = "offline"
    ONLINE = "online"


class ModelLoader(ForgeModel):
    """CoTracker3 model loader for video point tracking."""

    _VARIANTS = {
        ModelVariant.OFFLINE: ModelConfig(
            pretrained_model_name="cotracker3_offline",
        ),
        ModelVariant.ONLINE: ModelConfig(
            pretrained_model_name="cotracker3_online",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OFFLINE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize CoTracker3 model loader."""
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="CoTracker3",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_KEYPOINT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the CoTracker3 model instance."""
        model_name = self._variant_config.pretrained_model_name
        model = torch.hub.load("facebookresearch/co-tracker", model_name, **kwargs)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return input tensors for CoTracker3.

        Returns a synthetic video tensor of shape (B, T, C, H, W).
        """
        # Create a small synthetic video: 8 frames of 224x224 RGB
        video = torch.randn(batch_size, 8, 3, 224, 224)

        if dtype_override is not None:
            video = video.to(dtype_override)

        return video
