# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
AudioX-North-v1 model loader implementation for speech recognition (ASR) using PyTorch.
"""

import numpy as np
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
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
    """Available AudioX-North PyTorch speech recognition model variants."""

    AUDIOX_NORTH_V1 = "AudioX_North_v1"


class ModelLoader(ForgeModel):
    """AudioX-North-v1 model loader implementation for speech recognition (PyTorch)."""

    _VARIANTS = {
        ModelVariant.AUDIOX_NORTH_V1: ModelConfig(
            pretrained_model_name="jiviai/audioX-north-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.AUDIOX_NORTH_V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="AudioX_North",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model.config.forced_decoder_ids = None
        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)

        return self.model

    def load_inputs(self, dtype_override=None):
        if self.model is None or self.processor is None:
            self.load_model(dtype_override=dtype_override)

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate a synthetic 1-second audio waveform at 16kHz
        sampling_rate = 16000
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        input_features = self.processor(
            audio_array, sampling_rate=sampling_rate, return_tensors="pt"
        ).input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.tensor(
            [[self.model.config.decoder_start_token_id]],
            dtype=torch.long,
            device=device,
        )

        return [input_features, decoder_input_ids]
