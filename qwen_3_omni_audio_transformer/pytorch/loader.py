# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3-Omni AudioTransformer model loader implementation for audio feature extraction.

This model is the audio encoder component extracted from Qwen3-Omni-30B-A3B-Instruct.
It is a Whisper-style transformer encoder that produces audio embeddings from raw
audio waveforms.
"""

from typing import Optional

import numpy as np
import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen3-Omni AudioTransformer model variants."""

    DEFAULT = "default"


class Qwen3OmniAudioEncoderWrapper(torch.nn.Module):
    """Wrapper around Qwen3OmniMoeAudioEncoder for a clean forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        return self.model(input_features)


class ModelLoader(ForgeModel):
    """Qwen3-Omni AudioTransformer model loader for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="Atotti/Qwen3-Omni-AudioTransformer",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen3_Omni_AudioTransformer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import WhisperFeatureExtractor

        self._processor = WhisperFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3-Omni AudioTransformer model."""
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return Qwen3OmniAudioEncoderWrapper(model)

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen3-Omni AudioTransformer."""
        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz
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

        return [inputs["input_features"]]
