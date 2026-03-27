# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FSMN-VAD model loader implementation for voice activity detection.
"""

import torch
import torch.nn as nn
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


class FsmnVADWrapper(nn.Module):
    """Wrapper around the FSMN-VAD encoder to expose a standard forward interface."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats):
        return self.encoder(feats)


class ModelVariant(StrEnum):
    """Available FSMN-VAD model variants."""

    FSMN_VAD = "FSMN_VAD"


class ModelLoader(ForgeModel):
    """FSMN-VAD model loader implementation for voice activity detection."""

    _VARIANTS = {
        ModelVariant.FSMN_VAD: ModelConfig(
            pretrained_model_name="funasr/fsmn-vad",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FSMN_VAD

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._funasr_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FSMN-VAD",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from funasr import AutoModel

        self._funasr_model = AutoModel(
            model=self._variant_config.pretrained_model_name,
            model_revision="v2.0.4",
        )

        encoder = self._funasr_model.model.encoder
        model = FsmnVADWrapper(encoder)
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

        # Extract features using the FunASR frontend
        frontend = self._funasr_model.kwargs.get("frontend")
        feats, feats_len = frontend.extract_fbank(
            torch.tensor(audio_array).unsqueeze(0),
            torch.tensor([len(audio_array)]),
        )

        if dtype_override is not None:
            feats = feats.to(dtype_override)

        return (feats,)
