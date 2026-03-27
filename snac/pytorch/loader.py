# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SNAC (Multi-Scale Neural Audio Codec) model loader implementation.
"""

import torch
import numpy as np
from snac import SNAC
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
    """Available SNAC model variants."""

    SNAC_24KHZ = "SNAC 24kHz"


class ModelLoader(ForgeModel):
    """SNAC neural audio codec model loader implementation."""

    _VARIANTS = {
        ModelVariant.SNAC_24KHZ: ModelConfig(
            pretrained_model_name="hubertsiuzdak/snac_24khz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SNAC_24KHZ

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SNAC",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SNAC model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = SNAC.from_pretrained(pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the SNAC model."""
        # Generate a synthetic 1-second audio waveform at 24kHz
        # SNAC expects input shape (batch, channels=1, time_samples)
        sampling_rate = 24000
        duration_seconds = 1
        audio = torch.from_numpy(
            np.random.randn(1, 1, sampling_rate * duration_seconds).astype(np.float32)
        )

        if dtype_override is not None:
            audio = audio.to(dtype_override)

        return audio
