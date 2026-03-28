# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Fun-ASR-Nano model loader implementation for speech recognition (ASR).

Fun-ASR-Nano-2512 is an end-to-end speech recognition model from Tongyi Lab's
FunAudioLLM initiative, combining a SenseVoice audio encoder with a Qwen3
decoder. It supports 31 languages and achieves strong performance on
far-field and noisy speech recognition tasks.
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
    """Available Fun-ASR-Nano speech recognition model variants."""

    V2512_VLLM = "2512_vllm"


class FunASRNanoWrapper(torch.nn.Module):
    """Wrapper around the Fun-ASR-Nano model for a clean forward pass."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features):
        return self.model.inference(data_in=input_features)


class ModelLoader(ForgeModel):
    """Fun-ASR-Nano model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.V2512_VLLM: ModelConfig(
            pretrained_model_name="allendou/Fun-ASR-Nano-2512-vllm",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V2512_VLLM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._funasr_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Fun_ASR_Nano",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_funasr_model(self, dtype_override=None):
        """Load the Fun-ASR model using the funasr package."""
        from funasr import AutoModel

        model_kwargs = {
            "model": self._variant_config.pretrained_model_name,
            "trust_remote_code": True,
            "device": "cpu",
        }

        self._funasr_model = AutoModel(**model_kwargs)

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Fun-ASR-Nano model instance."""
        if self._funasr_model is None:
            self._load_funasr_model(dtype_override=dtype_override)

        model = FunASRNanoWrapper(self._funasr_model)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Fun-ASR-Nano model."""
        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        return [audio_array]
