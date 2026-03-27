# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XCodec2 neural speech codec model loader implementation.
"""

import torch
import numpy as np
from xcodec2.modeling_xcodec2 import XCodec2Model
from typing import Optional

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
    """Available XCodec2 model variants."""

    XCODEC2 = "XCodec2"


class ModelLoader(ForgeModel):
    """XCodec2 neural speech codec model loader implementation."""

    _VARIANTS = {
        ModelVariant.XCODEC2: ModelConfig(
            pretrained_model_name="HKUSTAudio/xcodec2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.XCODEC2

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="XCodec2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the XCodec2 model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = XCodec2Model.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the XCodec2 model."""
        # Generate a synthetic 1-second audio waveform at 16kHz
        # XCodec2 expects input shape (batch, time_samples)
        sampling_rate = 16000
        duration_seconds = 1
        audio = torch.from_numpy(
            np.random.randn(1, sampling_rate * duration_seconds).astype(np.float32)
        )

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return audio
