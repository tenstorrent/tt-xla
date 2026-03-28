# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Quran model loader implementation for speech recognition (ASR).

tarteel-ai/whisper-base-ar-quran is a fine-tuned version of openai/whisper-base
for Arabic Quranic speech recognition.
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperConfig,
)
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ....tools.utils import get_file
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available Whisper Quran model variants."""

    BASE_AR_QURAN = "Base_Ar_Quran"


class ModelLoader(ForgeModel):
    """Whisper Quran model loader implementation for speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.BASE_AR_QURAN: ModelConfig(
            pretrained_model_name="tarteel-ai/whisper-base-ar-quran",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_AR_QURAN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper_Quran",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Whisper Quran model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(pretrained_model_name)

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Whisper Quran model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        # Load audio sample
        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]
        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Preprocess audio
        sampling_rate = 16000
        processor_output = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = processor_output.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
