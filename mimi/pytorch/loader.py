# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mimi neural audio codec model loader implementation.
"""

import numpy as np
from transformers import MimiModel, AutoFeatureExtractor
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
    """Available Mimi model variants."""

    MIMI = "Mimi"


class ModelLoader(ForgeModel):
    """Mimi neural audio codec model loader implementation."""

    _VARIANTS = {
        ModelVariant.MIMI: ModelConfig(
            pretrained_model_name="kyutai/mimi",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MIMI

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.feature_extractor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Mimi",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_feature_extractor(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.feature_extractor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Mimi model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MimiModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the Mimi model."""
        if self.feature_extractor is None:
            self._load_feature_extractor()

        # Generate a synthetic 1-second audio waveform at 24kHz
        sampling_rate = self.feature_extractor.sampling_rate
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self.feature_extractor(
            raw_audio=audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["input_values"] = inputs["input_values"].to(dtype_override)

        return inputs["input_values"]
