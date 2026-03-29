# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PhoWhisper model loader implementation for Vietnamese automatic speech recognition (ASR).
"""

import torch
import numpy as np
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


class ModelVariant(StrEnum):
    """Available PhoWhisper model variants."""

    MEDIUM = "Medium"


class ModelLoader(ForgeModel):
    """PhoWhisper model loader implementation for Vietnamese ASR."""

    _VARIANTS = {
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="vinai/PhoWhisper-medium",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MEDIUM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="PhoWhisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import WhisperForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = WhisperForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name,
            use_cache=False,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        self._model = model
        return model

    def _load_processor(self):
        from transformers import WhisperProcessor

        self._processor = WhisperProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_inputs(self, dtype_override=None):
        from transformers import WhisperConfig

        if self._processor is None:
            self._load_processor()
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        # Generate synthetic 1-second audio at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        model_param = next(self._model.parameters())
        device = model_param.device
        dtype = dtype_override or model_param.dtype

        inputs = self._processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(device=device, dtype=dtype)

        whisper_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        decoder_input_ids = torch.full(
            (1, 2),
            whisper_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        return [input_features, decoder_input_ids]
