# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESPnet ECAPA-TDNN WavLM joint speaker embedding model loader.
"""

from typing import Optional

import torch

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


class ModelVariant(StrEnum):
    """Available ESPnet ECAPA-TDNN WavLM model variants."""

    ECAPA_WAVLM_JOINT = "ECAPA_WavLM_Joint"


class ModelLoader(ForgeModel):
    """ESPnet ECAPA-TDNN WavLM joint speaker embedding model loader."""

    _VARIANTS = {
        ModelVariant.ECAPA_WAVLM_JOINT: ModelConfig(
            pretrained_model_name="espnet/voxcelebs12_ecapa_wavlm_joint",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ECAPA_WAVLM_JOINT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="ESPnetECAPAWavLMVoxCeleb",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the ESPnet ECAPA-TDNN WavLM joint speaker embedding model."""
        from espnet2.bin.spk_inference import Speech2Embedding

        speech2embed = Speech2Embedding.from_pretrained(
            model_tag=self._variant_config.pretrained_model_name, **kwargs
        )
        model = speech2embed.spk_model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Generate sample audio input for the ECAPA-TDNN WavLM model.

        Returns a raw waveform tensor representing 1 second of 16kHz audio.
        """
        # The ESPnet speaker model expects raw waveform input
        # Shape: (batch, samples) - 1 second of 16kHz audio
        waveform = torch.randn(1, 16000)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return [waveform]
