#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
InfiniteTalk audio encoder loader implementation.

InfiniteTalk (MeiGen-AI/InfiniteTalk) is an audio-driven video dubbing
framework that generates lip-synced videos with natural head, body, and
facial expression alignment. The full pipeline is built on Wan2.1-I2V-14B
with a Wav2Vec2 audio encoder for audio conditioning.

This loader tests the Wav2Vec2 audio encoder component
(TencentGameMate/chinese-wav2vec2-base) used for audio conditioning
in the InfiniteTalk pipeline.
"""

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
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
    """Available InfiniteTalk model variants."""

    SINGLE = "single"


class ModelLoader(ForgeModel):
    """InfiniteTalk audio encoder loader for audio-driven video dubbing."""

    _VARIANTS = {
        ModelVariant.SINGLE: ModelConfig(
            pretrained_model_name="TencentGameMate/chinese-wav2vec2-base",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SINGLE

    # Audio encoder used for conditioning in the InfiniteTalk pipeline
    AUDIO_ENCODER_NAME = "TencentGameMate/chinese-wav2vec2-base"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize InfiniteTalk audio encoder loader."""
        super().__init__(variant)
        self._processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="InfiniteTalk",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_processor(self):
        """Load the Wav2Vec2 feature extractor for audio preprocessing."""
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Wav2Vec2 audio encoder used by InfiniteTalk."""
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Wav2Vec2Model.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        if self._processor is None:
            self._load_processor()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return synthetic audio inputs for the audio encoder."""
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

        if dtype_override is not None:
            inputs = {
                k: v.to(dtype_override) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        return dict(inputs)
