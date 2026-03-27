# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Optimum Intel Internal Testing tiny-random-whisper model loader implementation
for speech recognition (ASR).
"""

import numpy as np
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)
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
    """Available tiny-random-whisper model variants."""

    TINY_RANDOM_WHISPER = "tiny_random_whisper"


class ModelLoader(ForgeModel):
    """Optimum Intel Internal Testing tiny-random-whisper model loader for ASR."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM_WHISPER: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-whisper",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM_WHISPER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Optimum_Intel_Internal_Testing",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["torch_dtype"] = dtype_override

        self._processor = WhisperProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate synthetic 30-second audio at 16kHz to match Whisper's receptive field
        sampling_rate = 16000
        duration_seconds = 30
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        whisper_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        inputs = self._processor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        device = torch.device("cpu")
        dtype = dtype_override or torch.float32

        input_features = inputs.input_features.to(device=device, dtype=dtype)
        decoder_input_ids = torch.full(
            (1, 1),
            whisper_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        return [input_features, decoder_input_ids]
