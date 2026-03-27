# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3 ASR model loader implementation for speech recognition (ASR) using PyTorch.
"""

import torch
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
    """Available Qwen3 ASR PyTorch speech recognition model variants."""

    QWEN3_ASR_1_7B = "1_7B"


class ModelLoader(ForgeModel):
    """Qwen3 ASR model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.QWEN3_ASR_1_7B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-ASR-1.7B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_ASR_1_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen3 ASR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name, fix_mistral_regex=True
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        # Importing qwen_asr registers the qwen3_asr architecture with transformers
        import qwen_asr  # noqa: F401
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        import numpy as np

        if self._processor is None:
            self._load_processor()

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        # Build the text prompt expected by the Qwen3 ASR processor
        text = (
            "<|startoftext|><|im_start|>system\n"
            "You are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        inputs = self._processor(
            text=text,
            audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key in inputs:
                if torch.is_tensor(inputs[key]) and inputs[key].is_floating_point():
                    inputs[key] = inputs[key].to(dtype_override)

        return inputs
