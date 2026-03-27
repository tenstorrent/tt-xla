# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MP-SENet speech enhancement model loader implementation.

MP-SENet performs magnitude and phase speech enhancement in parallel,
denoising audio waveforms using a conformer-based architecture.
"""

import torch
import numpy as np
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
    """Available MP-SENet model variants."""

    DNS = "DNS"


class ModelLoader(ForgeModel):
    """MP-SENet speech enhancement model loader implementation."""

    _VARIANTS = {
        ModelVariant.DNS: ModelConfig(
            pretrained_model_name="JacobLinCool/MP-SENet-DNS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DNS

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MP-SENet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from MPSENet import MPSENet

        model = MPSENet.from_pretrained(self._variant_config.pretrained_model_name)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio = np.random.randn(sampling_rate * duration_seconds).astype(np.float32)

        audio_tensor = torch.from_numpy(audio)

        if dtype_override is not None:
            audio_tensor = audio_tensor.to(dtype_override)

        return audio_tensor
