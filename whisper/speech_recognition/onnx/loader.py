# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper ONNX model loader for speech recognition (ASR).
"""

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from transformers import WhisperProcessor

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
from typing import Optional


class ModelVariant(StrEnum):
    """Available Whisper ONNX speech recognition model variants."""

    BASE_EN = "Base_en"


class ModelLoader(ForgeModel):
    """Whisper ONNX model loader for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.BASE_EN: ModelConfig(
            pretrained_model_name="Xenova/whisper-base.en",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_EN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.ONNX,
        )

    def load_model(self, **kwargs):
        """Download and load the Whisper encoder ONNX model from Hugging Face.

        Returns:
            onnx.ModelProto: The loaded ONNX encoder model.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        encoder_path = hf_hub_download(
            pretrained_model_name, filename="onnx/encoder_model.onnx"
        )
        model = onnx.load(encoder_path)

        return model

    def load_inputs(self, **kwargs):
        """Generate sample inputs for the Whisper ONNX encoder model.

        Returns:
            numpy.ndarray: Input features array suitable for the ONNX encoder.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._processor is None:
            self._processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        # Generate synthetic 1-second audio at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np",
        )

        return inputs.input_features
