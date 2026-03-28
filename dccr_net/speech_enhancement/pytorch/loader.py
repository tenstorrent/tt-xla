# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DCCRNet model loader implementation for speech enhancement using PyTorch.
"""

import torch
from typing import Optional

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
    """Available DCCRNet PyTorch speech enhancement model variants."""

    LIBRI1MIX_ENH_16K = "Libri1Mix_enhsingle_16k"


class ModelLoader(ForgeModel):
    """DCCRNet model loader implementation for speech enhancement (PyTorch)."""

    _VARIANTS = {
        ModelVariant.LIBRI1MIX_ENH_16K: ModelConfig(
            pretrained_model_name="JorisCos/DCCRNet_Libri1Mix_enhsingle_16k",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LIBRI1MIX_ENH_16K

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DCCRNet",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from asteroid.models import BaseModel

        model = BaseModel.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        # DCCRNet expects a batch of waveforms: (batch, time)
        waveform = torch.from_numpy(audio_array).unsqueeze(0)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return (waveform,)
