# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Voxtral Mini model loader implementation for real-time speech recognition (ASR).
"""

import numpy as np
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
    """Available Voxtral Mini model variants."""

    VOXTRAL_MINI_4B_REALTIME = "Voxtral_Mini_4B_Realtime"


class ModelLoader(ForgeModel):
    """Voxtral Mini model loader implementation for real-time ASR."""

    _VARIANTS = {
        ModelVariant.VOXTRAL_MINI_4B_REALTIME: ModelConfig(
            pretrained_model_name="mistralai/Voxtral-Mini-4B-Realtime-2602",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOXTRAL_MINI_4B_REALTIME

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._processor = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Voxtral-Mini",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import VoxtralRealtimeForConditionalGeneration

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = VoxtralRealtimeForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        if dtype_override is not None:
            model.to(dtype_override)

        self._model = model
        return model

    def _load_processor(self):
        from transformers import AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self._processor

    def load_inputs(self, dtype_override=None):
        import torch

        if self._processor is None:
            self._load_processor()
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        # Generate a synthetic 1-second audio waveform at the expected sampling rate
        sampling_rate = self._processor.feature_extractor.sampling_rate
        duration_seconds = 1
        audio_array = np.random.randn(sampling_rate * duration_seconds).astype(
            np.float32
        )

        inputs = self._processor(audio_array, return_tensors="pt")

        model_param = next(self._model.parameters())
        dtype = dtype_override or model_param.dtype
        device = model_param.device

        # Pre-compute encoder_inputs_embeds from input_features, matching the
        # generate() pipeline in _prepare_model_inputs. The raw input_features
        # produce more audio tokens than input_ids, so we embed and slice to
        # align with the text sequence length.
        input_features = inputs.pop("input_features").to(device=device, dtype=dtype)
        seq_len = inputs["input_ids"].shape[1]
        downsample_factor = self._model.config.downsample_factor
        with torch.no_grad():
            encoder_inputs_embeds = self._model.audio_tower.embedder(input_features)
        encoder_inputs_embeds = encoder_inputs_embeds[
            :, : seq_len * downsample_factor, :
        ]
        inputs["encoder_inputs_embeds"] = encoder_inputs_embeds

        return inputs
