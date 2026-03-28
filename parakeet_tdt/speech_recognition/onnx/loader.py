# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Parakeet TDT ONNX model loader implementation for speech recognition (ASR).
"""

import torch
import onnx
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
    """Available Parakeet TDT ONNX speech recognition model variants."""

    PARAKEET_TDT_0_6B_V3 = "Parakeet_TDT_0.6B_v3_ONNX"


class ModelLoader(ForgeModel):
    """Parakeet TDT ONNX model loader implementation for speech recognition."""

    _VARIANTS = {
        ModelVariant.PARAKEET_TDT_0_6B_V3: ModelConfig(
            pretrained_model_name="istupakov/parakeet-tdt-0.6b-v3-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PARAKEET_TDT_0_6B_V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Parakeet_TDT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        from huggingface_hub import hf_hub_download

        pretrained = self._variant_config.pretrained_model_name
        encoder_path = hf_hub_download(pretrained, "encoder-model.onnx")
        # Also download the external data file required by the encoder
        hf_hub_download(pretrained, "encoder-model.onnx.data")

        model = onnx.load(encoder_path)
        return model

    def load_inputs(self, **kwargs):
        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_signal = torch.randn(1, sampling_rate * duration_seconds)
        length = torch.tensor([sampling_rate * duration_seconds], dtype=torch.long)

        return {"audio_signal": audio_signal, "length": length}
