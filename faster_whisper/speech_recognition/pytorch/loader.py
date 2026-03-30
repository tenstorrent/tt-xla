# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Faster Whisper model loader implementation for speech recognition (ASR).

Note: Faster Whisper models are CTranslate2-quantized versions of Whisper models.
Since CTranslate2 format is not compatible with PyTorch, this loader uses the base
Whisper models via WhisperForConditionalGeneration.

deepdml/faster-distil-whisper-large-v3.5 is a CTranslate2-quantized version of
distil-whisper/distil-large-v3.5.

Systran/faster-whisper-large-v1 is a CTranslate2-quantized version of
openai/whisper-large.

Systran/faster-distil-whisper-small.en is a CTranslate2-quantized version of
distil-whisper/distil-small.en.
"""

from typing import Optional

import numpy as np
import torch

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
    """Available Faster Whisper speech recognition model variants."""

    DISTIL_LARGE_V3_5 = "Distil_Large_v3_5"
    DISTIL_SMALL_EN = "Distil_Small_en"
    LARGE_V1 = "Large_v1"


class ModelLoader(ForgeModel):
    """Faster Whisper model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.DISTIL_LARGE_V3_5: ModelConfig(
            pretrained_model_name="distil-whisper/distil-large-v3.5",
        ),
        ModelVariant.DISTIL_SMALL_EN: ModelConfig(
            pretrained_model_name="distil-whisper/distil-small.en",
        ),
        ModelVariant.LARGE_V1: ModelConfig(
            pretrained_model_name="openai/whisper-large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DISTIL_LARGE_V3_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """

        super().__init__(variant)
        self._processor = None
        self._model = None
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
            model="Faster_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Whisper model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self._model = WhisperForConditionalGeneration.from_pretrained(
            self._model_name, use_cache=False, **model_kwargs
        )
        self._processor = WhisperProcessor.from_pretrained(self._model_name)

        self._model.eval()
        if dtype_override is not None:
            self._model.to(dtype_override)
        return self._model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Whisper model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from transformers import WhisperConfig

        if self._model is None or self._processor is None:
            self.load_model(dtype_override=dtype_override)

        whisper_config = WhisperConfig.from_pretrained(self._model_name)

        # Generate synthetic 30-second audio at 16kHz to match Whisper's receptive field
        sampling_rate = 16000
        duration_seconds = 30
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        model_param = next(self._model.parameters())
        device = model_param.device
        dtype = dtype_override or model_param.dtype

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        input_features = inputs.input_features.to(device=device, dtype=dtype)
        decoder_input_ids = torch.full(
            (1, 2),
            whisper_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        return [input_features, decoder_input_ids]
