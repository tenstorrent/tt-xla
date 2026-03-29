# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Distil-Whisper OpenVINO model loader implementation for speech recognition (ASR).

OpenVINO/distil-whisper-large-v3-int8-ov is a distilled version of OpenAI's Whisper
large-v3, converted to OpenVINO IR format with INT8 quantization using NNCF.
"""

from typing import Optional

import numpy as np

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
    """Available Distil-Whisper OpenVINO speech recognition model variants."""

    LARGE_V3_INT8 = "Large_v3_int8"


class ModelLoader(ForgeModel):
    """Distil-Whisper OpenVINO model loader implementation for speech recognition."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_INT8: ModelConfig(
            pretrained_model_name="OpenVINO/distil-whisper-large-v3-int8-ov",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_INT8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Distil_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq

        model = OVModelForSpeechSeq2Seq.from_pretrained(
            self._variant_config.pretrained_model_name, **kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        return inputs
