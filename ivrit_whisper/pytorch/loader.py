# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Ivrit Whisper model loader implementation for Hebrew speech recognition (ASR).

Note: The original model (ivrit-ai/whisper-large-v3-turbo-ct2) is in CTranslate2
format. Since CTranslate2 format is not compatible with PyTorch, this loader uses
the standard transformers-compatible version (ivrit-ai/whisper-large-v3-turbo) via
WhisperForConditionalGeneration.
"""

from typing import Optional

import torch

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Ivrit Whisper model variants."""

    LARGE_V3_TURBO = "Large_v3_Turbo"


class ModelLoader(ForgeModel):
    """Ivrit Whisper model loader implementation for Hebrew speech recognition (ASR)."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="ivrit-ai/whisper-large-v3-turbo",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_TURBO

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """

        super().__init__(variant)
        self.processor = None
        self.model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Ivrit_Whisper",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Ivrit Whisper model instance.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            self._model_name, use_cache=False, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(
            self._model_name, use_cache=False, **model_kwargs
        )

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Ivrit Whisper model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        from transformers import AutoProcessor, WhisperConfig

        if self.model is None or self.processor is None:
            self.load_model()

        whisper_config = WhisperConfig.from_pretrained(self._model_name)

        # Load audio sample
        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]
        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Preprocess audio using v3 turbo processor
        processor = AutoProcessor.from_pretrained(self._model_name)
        features = processor.feature_extractor(
            sample_audio,
            sampling_rate=processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            return_token_timestamps=True,
            return_attention_mask=True,
        )
        input_features = features["input_features"].to(device=device, dtype=dtype)
        attention_mask = features.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Build decoder input IDs for Hebrew transcription
        decoder_prompt_ids = self.processor.get_decoder_prompt_ids(
            task="transcribe", language="he", no_timestamps=True
        )
        init_tokens = [self.model.generation_config.decoder_start_token_id]
        if decoder_prompt_ids:
            init_tokens += [tok for _, tok in decoder_prompt_ids]

        decoder_input_ids = torch.tensor([init_tokens], dtype=torch.long, device=device)
        return [input_features, attention_mask, decoder_input_ids]
