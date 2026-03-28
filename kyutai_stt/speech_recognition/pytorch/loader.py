# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Kyutai STT model loader implementation for speech recognition (ASR) using PyTorch.
"""

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
    """Available Kyutai STT PyTorch speech recognition model variants."""

    STT_2_6B_EN = "STT_2.6B_EN"


class ModelLoader(ForgeModel):
    """Kyutai STT model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.STT_2_6B_EN: ModelConfig(
            pretrained_model_name="kyutai/stt-2.6b-en-trfs",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.STT_2_6B_EN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="KyutaiSTT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self, dtype_override=None):
        from transformers import KyutaiSpeechToTextProcessor

        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        self._processor = KyutaiSpeechToTextProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import KyutaiSpeechToTextForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = KyutaiSpeechToTextForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        # Generate a synthetic 3-second audio waveform at 24kHz
        sampling_rate = 24000
        duration_seconds = 3
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(
            audio_array,
            return_tensors="pt",
        )

        return inputs
