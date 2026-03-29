# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpatialTrackerV2 (VGGT4Track) model loader for 3D point tracking and depth estimation.
"""

import torch
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
from .src import VGGT4Track


class ModelVariant(StrEnum):
    """Available SpatialTrackerV2 model variants."""

    FRONT = "Front"


class ModelLoader(ForgeModel):
    """SpatialTrackerV2 Front model loader for 3D geometry prediction from video."""

    _VARIANTS = {
        ModelVariant.FRONT: ModelConfig(
            pretrained_model_name="Yuxihenry/SpatialTrackerV2_Front",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FRONT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpatialTrackerV2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_DEPTH_EST,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = VGGT4Track.from_pretrained(pretrained_model_name)

        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        dtype = dtype_override or torch.float32

        # Video input: [B, S, 3, H, W] - 2 frames of 518x518 images
        images = torch.randn(batch_size, 2, 3, 518, 518, dtype=dtype)
        # Clamp to [0, 1] range as expected by the model
        images = images.clamp(0, 1)

        return {"images": images}
