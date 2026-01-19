# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper model loader implementation
"""

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperModel,
    AutoFeatureExtractor,
    WhisperConfig,
    AutoProcessor,
)
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...tools.utils import get_file
from ...base import ForgeModel
from typing import Optional


class ModelVariant(StrEnum):
    """Available Whisper model variants."""

    WHISPER_TINY = "openai/whisper-tiny"
    WHISPER_BASE = "openai/whisper-base"
    WHISPER_SMALL = "openai/whisper-small"
    WHISPER_MEDIUM = "openai/whisper-medium"
    WHISPER_LARGE = "openai/whisper-large"
    WHISPER_LARGE_V3 = "openai/whisper-large-v3"
    WHISPER_LARGE_V3_TURBO = "openai/whisper-large-v3-turbo"


class ModelLoader(ForgeModel):
    """Whisper model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.WHISPER_TINY: ModelConfig(
            pretrained_model_name="openai/whisper-tiny",
        ),
        ModelVariant.WHISPER_BASE: ModelConfig(
            pretrained_model_name="openai/whisper-base",
        ),
        ModelVariant.WHISPER_SMALL: ModelConfig(
            pretrained_model_name="openai/whisper-small",
        ),
        ModelVariant.WHISPER_MEDIUM: ModelConfig(
            pretrained_model_name="openai/whisper-medium",
        ),
        ModelVariant.WHISPER_LARGE: ModelConfig(
            pretrained_model_name="openai/whisper-large",
        ),
        ModelVariant.WHISPER_LARGE_V3: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3",
        ),
        ModelVariant.WHISPER_LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3-turbo",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.WHISPER_TINY

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="whisper",
            variant=variant,
            group=(
                ModelGroup.RED
                if variant == ModelVariant.WHISPER_LARGE_V3
                else ModelGroup.GENERALITY
            ),
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.processor = None
        self.feature_extractor = None
        self.model = None

    def load_model(self, dtype_override=None):
        """Load a Whisper model from Hugging Face."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Common model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self._variant == ModelVariant.WHISPER_LARGE_V3:
            self.model = WhisperModel.from_pretrained(
                pretrained_model_name, **model_kwargs
            )
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                pretrained_model_name
            )
            self.processor = None
        else:
            self.model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name, use_cache=False, **model_kwargs
            )
            self.processor = WhisperProcessor.from_pretrained(
                pretrained_model_name, use_cache=False, **model_kwargs
            )
            self.feature_extractor = None

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Whisper model."""

        # Ensure model and pre-processing utilities are initialized
        if self.model is None or (
            self.processor is None and self.feature_extractor is None
        ):
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
        if hasattr(self, "feature_extractor") and self.feature_extractor is not None:
            processor = self.feature_extractor(
                sample_audio, return_tensors="pt", sampling_rate=sampling_rate
            )
        else:
            processor = self.processor(
                sample_audio, return_tensors="pt", sampling_rate=sampling_rate
            )

        input_features = processor.input_features.to(device=device, dtype=dtype)

        if self._variant == ModelVariant.WHISPER_LARGE_V3_TURBO:
            processor_v3 = AutoProcessor.from_pretrained(
                self._variant_config.pretrained_model_name
            )
            features_v3 = processor_v3.feature_extractor(
                sample_audio,
                sampling_rate=processor_v3.feature_extractor.sampling_rate,
                return_tensors="pt",
                return_token_timestamps=True,
                return_attention_mask=True,
            )
            input_features = features_v3["input_features"].to(
                device=device, dtype=dtype
            )
            attention_mask = features_v3.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Build decoder input IDs
            decoder_prompt_ids = self.processor.get_decoder_prompt_ids(
                task="transcribe", language="en", no_timestamps=True
            )
            init_tokens = [self.model.generation_config.decoder_start_token_id]
            if decoder_prompt_ids:
                init_tokens += [tok for _, tok in decoder_prompt_ids]

            decoder_input_ids = torch.tensor(
                [init_tokens], dtype=torch.long, device=device
            )
            return [input_features, attention_mask, decoder_input_ids]

        decoder_input_ids = torch.full(
            (1, 2), model_config.decoder_start_token_id, dtype=torch.long, device=device
        )
        return [input_features, decoder_input_ids]
