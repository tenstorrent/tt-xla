# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper Base MLX model loader implementation for speech recognition (ASR).

MLX community conversion of OpenAI Whisper base model,
an encoder-decoder Transformer for automatic speech recognition.
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
    """Available Whisper Base MLX speech recognition model variants."""

    WHISPER_BASE_MLX = "Whisper_Base_MLX"


class ModelLoader(ForgeModel):
    """Whisper Base MLX model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.WHISPER_BASE_MLX: ModelConfig(
            pretrained_model_name="mlx-community/whisper-base-mlx",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WHISPER_BASE_MLX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self._model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper_Base_MLX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
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
