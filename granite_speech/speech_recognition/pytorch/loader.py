# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Granite Speech model loader implementation for speech recognition (ASR) using PyTorch.

Granite-4.0-1b-speech is a compact encoder-decoder speech-language model from IBM
supporting multilingual ASR and bidirectional speech translation across 6 languages.
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
    """Available Granite Speech PyTorch model variants."""

    V4_0_1B = "4.0-1b"


class ModelLoader(ForgeModel):
    """Granite Speech model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.V4_0_1B: ModelConfig(
            pretrained_model_name="ibm-granite/granite-4.0-1b-speech",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V4_0_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Granite_Speech",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSpeechSeq2Seq

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        user_prompt = "<|audio|>can you transcribe the speech into a written format?"
        chat = [{"role": "user", "content": user_prompt}]
        prompt = self._processor.tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            prompt, audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        )

        return inputs
