# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Parakeet TDT ONNX model loader implementation for speech recognition (ASR).
"""

from typing import Optional

import numpy as np
import onnx

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Parakeet TDT ONNX speech recognition model variants."""

    PARAKEET_TDT_0_6B_V2 = "Parakeet_TDT_0.6B_v2"


class ModelLoader(ForgeModel):
    """Parakeet TDT ONNX model loader implementation for speech recognition."""

    _VARIANTS = {
        ModelVariant.PARAKEET_TDT_0_6B_V2: ModelConfig(
            pretrained_model_name="istupakov/parakeet-tdt-0.6b-v2-onnx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PARAKEET_TDT_0_6B_V2

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

        local_path = hf_hub_download(
            repo_id=self._variant_config.pretrained_model_name,
            filename="encoder-model.onnx",
        )
        model = onnx.load(local_path)

        return model

    def load_inputs(self, **kwargs):
        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(1, sampling_rate * duration_seconds).astype(
            np.float32
        )
        length_array = np.array([sampling_rate * duration_seconds], dtype=np.int64)

        return {
            "audio_signal": audio_array,
            "length": length_array,
        }
