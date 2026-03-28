# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EnCodec neural audio codec model loader implementation.
"""

import numpy as np
from transformers import EncodecModel, AutoProcessor
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
    """Available EnCodec model variants."""

    ENCODEC_48KHZ = "EnCodec 48kHz"


class ModelLoader(ForgeModel):
    """EnCodec neural audio codec model loader implementation."""

    _VARIANTS = {
        ModelVariant.ENCODEC_48KHZ: ModelConfig(
            pretrained_model_name="facebook/encodec_48khz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ENCODEC_48KHZ

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="EnCodec",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the EnCodec model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.processor = AutoProcessor.from_pretrained(pretrained_model_name)
        model = EncodecModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample audio inputs for the EnCodec model."""
        if not hasattr(self, "processor") or self.processor is None:
            self.load_model()

        # Generate a synthetic 1-second stereo audio waveform at 48kHz
        sampling_rate = 48000
        duration_seconds = 1
        audio = np.random.randn(2, sampling_rate * duration_seconds).astype(np.float32)

        inputs = self.processor(
            raw_audio=audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )

        if dtype_override is not None:
            inputs["input_values"] = inputs["input_values"].to(dtype_override)

        return inputs
