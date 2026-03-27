# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-ASR model loader implementation for speech recognition (ASR).

Qwen3-ASR-0.6B is a lightweight automatic speech recognition model from
Alibaba's Qwen team, supporting multilingual speech-to-text conversion.
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
    """Available Qwen3-ASR speech recognition model variants."""

    V0_6B = "0.6B"


class Qwen3ASRWrapper(torch.nn.Module):
    """Wrapper around the Qwen3-ASR thinker model for a clean forward pass."""

    def __init__(self, thinker):
        super().__init__()
        self.thinker = thinker

    def forward(
        self, input_ids, attention_mask, input_features, feature_attention_mask
    ):
        return self.thinker(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )


class ModelLoader(ForgeModel):
    """Qwen3-ASR model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.V0_6B: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-ASR-0.6B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V0_6B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self._model_wrapper = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Qwen3_ASR",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_model_wrapper(self, dtype_override=None):
        """Load the qwen_asr model wrapper and cache processor."""
        from qwen_asr import Qwen3ASRModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        else:
            model_kwargs["dtype"] = torch.float32

        self._model_wrapper = Qwen3ASRModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            device_map="cpu",
            max_new_tokens=50,
            **model_kwargs,
        )
        self._processor = self._model_wrapper.processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen3-ASR model instance."""
        if self._model_wrapper is None:
            self._load_model_wrapper(dtype_override=dtype_override)

        thinker = self._model_wrapper.model.thinker
        model = Qwen3ASRWrapper(thinker)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Qwen3-ASR model."""
        if self._model_wrapper is None:
            self._load_model_wrapper(dtype_override=dtype_override)

        text = self._model_wrapper._build_text_prompt(
            context="", force_language="English"
        )
        audio = np.random.randn(16000).astype(np.float32)

        inputs = self._processor(
            text=[text], audio=[audio], return_tensors="pt", padding=True
        )

        return [
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["input_features"],
            inputs["feature_attention_mask"],
        ]
