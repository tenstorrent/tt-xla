# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Faster Whisper model loader implementation for speech recognition (ASR).

Note: Faster Whisper models are CTranslate2-quantized versions of OpenAI's
Whisper models. Since CTranslate2 format is not compatible with PyTorch,
this loader uses the base OpenAI Whisper models via
WhisperForConditionalGeneration.
"""

from typing import Optional

import numpy as np
import torch
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

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
    """Available Faster Whisper speech recognition model variants."""

    TINY_INT8 = "Tiny_int8"
    LARGE_V2 = "Large_v2"
    LARGE_V3_TURBO = "Large_v3_Turbo"


class ModelLoader(ForgeModel):
    """Faster Whisper PyTorch model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.TINY_INT8: ModelConfig(
            pretrained_model_name="openai/whisper-tiny",
        ),
        ModelVariant.LARGE_V2: ModelConfig(
            pretrained_model_name="openai/whisper-large-v2",
        ),
        ModelVariant.LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3-turbo",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_INT8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Faster_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Faster Whisper model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Faster Whisper model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic 30-second audio at 16kHz to match Whisper's receptive field
        sample_audio = np.random.randn(16000 * 30).astype(np.float32)
        processor_output = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=16000
        )
        input_features = processor_output.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
