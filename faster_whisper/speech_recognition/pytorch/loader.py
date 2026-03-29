# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Faster Whisper model loader implementation for speech recognition (ASR) using PyTorch.

Note: Faster Whisper models are CTranslate2-quantized versions of OpenAI's
Whisper models. Since CTranslate2 format is not compatible with PyTorch,
this loader uses the base OpenAI Whisper models via
WhisperForConditionalGeneration.
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
    """Available Faster Whisper PyTorch speech recognition model variants."""

    TINY = "Tiny"


class ModelLoader(ForgeModel):
    """Faster Whisper model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.TINY: ModelConfig(
            pretrained_model_name="openai/whisper-tiny",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
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

    def _load_processor(self, dtype_override=None):
        from transformers import WhisperProcessor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = WhisperProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

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

        return model

    def load_inputs(self, dtype_override=None):
        from transformers import WhisperConfig

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

        inputs["decoder_input_ids"] = torch.full(
            (1, 2), whisper_config.decoder_start_token_id, dtype=torch.long
        )

        return inputs
