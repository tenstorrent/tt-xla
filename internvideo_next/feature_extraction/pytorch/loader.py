# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
InternVideoNext feature extraction model loader implementation for PyTorch.

InternVideoNext is a video foundation model for extracting video features/embeddings.
It processes video frames using a Vision Transformer architecture with patch size 14
and produces dense feature representations.
"""

from typing import Optional

import torch
from transformers import AutoModel, VideoMAEImageProcessor

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available InternVideoNext feature extraction model variants."""

    LARGE_P14_RES224_F16 = "Large_P14_Res224_F16"


class ModelLoader(ForgeModel):
    """InternVideoNext feature extraction model loader implementation for PyTorch."""

    _VARIANTS = {
        ModelVariant.LARGE_P14_RES224_F16: ModelConfig(
            pretrained_model_name="revliter/internvideo_next_large_p14_res224_f16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_P14_RES224_F16

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="InternVideoNext",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_VIDEO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the InternVideoNext feature extraction model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            model: The loaded InternVideoNext model instance
        """
        model = AutoModel.from_pretrained(
            self._model_name, trust_remote_code=True, **kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model.

        Generates synthetic video input matching the expected format:
        (batch, channels, frames, height, width) with 16 frames at 224x224 resolution.

        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).

        Returns:
            dict: Input tensors with pixel_values for video frames.
        """
        num_frames = 16
        height = 224
        width = 224
        channels = 3

        # Model expects input shape: (B, C, T, H, W)
        pixel_values = torch.randn(batch_size, channels, num_frames, height, width)

        if dtype_override is not None:
            pixel_values = pixel_values.to(dtype_override)

        return {"pixel_values": pixel_values}
