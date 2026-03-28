# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Orange Speaker-wavLM-tbr model loader implementation for speaker embeddings.
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
    """Available Speaker-wavLM-tbr model variants."""

    SPEAKER_WAVLM_TBR = "Speaker-wavLM-tbr"


class ModelLoader(ForgeModel):
    """Orange Speaker-wavLM-tbr model loader implementation for speaker timbral embeddings."""

    _VARIANTS = {
        ModelVariant.SPEAKER_WAVLM_TBR: ModelConfig(
            pretrained_model_name="Orange/Speaker-wavLM-tbr",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPEAKER_WAVLM_TBR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SpeakerWavLMTbr",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **kwargs,
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        # Model expects raw 16kHz mono audio waveform
        # Generate a synthetic 1-second waveform
        waveform = torch.randn(1, 16000)

        if dtype_override is not None:
            waveform = waveform.to(dtype_override)

        return [waveform]
