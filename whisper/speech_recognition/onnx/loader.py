# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper ONNX model loader implementation for speech recognition (ASR).
"""

from typing import Optional

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


class ModelVariant(StrEnum):
    """Available Whisper ONNX speech recognition model variants."""

    BASE = "Base"


class ModelLoader(ForgeModel):
    """Whisper ONNX model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="onnx-community/whisper-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._processor = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
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
        """Load and return the Whisper ONNX encoder model.

        Returns:
            onnx.ModelProto: The loaded ONNX encoder model.
        """
        encoder_path = hf_hub_download(
            self._model_name, filename="onnx/encoder_model.onnx"
        )
        model = onnx.load(encoder_path)
        return model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Whisper ONNX encoder model.

        Returns:
            numpy.ndarray: Input features array for the encoder.
        """
        if self._processor is None:
            self._processor = WhisperProcessor.from_pretrained(self._model_name)

        # Generate synthetic 30-second audio at 16kHz
        sampling_rate = 16000
        duration_seconds = 30
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="np",
        )

        return inputs.input_features
