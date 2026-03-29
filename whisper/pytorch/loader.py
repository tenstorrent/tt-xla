# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper model loader implementation
"""

import numpy as np
import torch
from huggingface_hub import hf_hub_download
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

    WHISPER_TINY = "Tiny"
    WHISPER_TINY_EN = "Tiny_en"
    WHISPER_BASE = "Base"
    WHISPER_BASE_EN = "Base_en"
    WHISPER_SMALL = "Small"
    WHISPER_SMALL_EN = "Small_en"
    WHISPER_MEDIUM = "Medium"
    WHISPER_MEDIUM_EN = "Medium_en"
    WHISPER_LARGE = "Large"
    WHISPER_LARGE_V2 = "Large_v2"
    WHISPER_LARGE_V3 = "Large_v3"
    WHISPER_LARGE_V3_TURBO = "Large_v3_Turbo"
    WHISPER_MEDIUM_JP = "Medium_jp"
    WHISPER_BASE_BUNGOMA_EN = "Base_Bungoma_en"


class ModelLoader(ForgeModel):
    """Whisper model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.WHISPER_TINY: ModelConfig(
            pretrained_model_name="openai/whisper-tiny",
        ),
        ModelVariant.WHISPER_TINY_EN: ModelConfig(
            pretrained_model_name="openai/whisper-tiny.en",
        ),
        ModelVariant.WHISPER_BASE: ModelConfig(
            pretrained_model_name="openai/whisper-base",
        ),
        ModelVariant.WHISPER_BASE_EN: ModelConfig(
            pretrained_model_name="openai/whisper-base.en",
        ),
        ModelVariant.WHISPER_SMALL: ModelConfig(
            pretrained_model_name="openai/whisper-small",
        ),
        ModelVariant.WHISPER_SMALL_EN: ModelConfig(
            pretrained_model_name="openai/whisper-small.en",
        ),
        ModelVariant.WHISPER_MEDIUM: ModelConfig(
            pretrained_model_name="openai/whisper-medium",
        ),
        ModelVariant.WHISPER_MEDIUM_EN: ModelConfig(
            pretrained_model_name="openai/whisper-medium.en",
        ),
        ModelVariant.WHISPER_LARGE: ModelConfig(
            pretrained_model_name="openai/whisper-large",
        ),
        ModelVariant.WHISPER_LARGE_V2: ModelConfig(
            pretrained_model_name="openai/whisper-large-v2",
        ),
        ModelVariant.WHISPER_LARGE_V3: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3",
        ),
        ModelVariant.WHISPER_LARGE_V3_TURBO: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3-turbo",
        ),
        ModelVariant.WHISPER_MEDIUM_JP: ModelConfig(
            pretrained_model_name="vumichien/whisper-medium-jp",
        ),
        ModelVariant.WHISPER_BASE_BUNGOMA_EN: ModelConfig(
            pretrained_model_name="eai6/whisper-base-bungoma.en",
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
        if variant == ModelVariant.WHISPER_TINY_EN:
            group = ModelGroup.VULCAN
        elif variant == ModelVariant.WHISPER_LARGE_V3:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="Whisper",
            variant=variant,
            group=(
                ModelGroup.RED
                if variant == ModelVariant.WHISPER_LARGE_V3
                else ModelGroup.VULCAN
                if variant
                in (
                    ModelVariant.WHISPER_MEDIUM_EN,
                    ModelVariant.WHISPER_MEDIUM_JP,
                    ModelVariant.WHISPER_BASE_BUNGOMA_EN,
                )
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

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load a Whisper model from Hugging Face."""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Common model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self._variant == ModelVariant.WHISPER_MEDIUM_MLX:
            self.model = self._load_mlx_model(pretrained_model_name, **model_kwargs)
            self.processor = WhisperProcessor.from_pretrained(
                "openai/whisper-medium", use_cache=False
            )
            self.feature_extractor = None
        elif self._variant == ModelVariant.WHISPER_LARGE_V3:
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

    @staticmethod
    def _load_mlx_model(pretrained_model_name, **model_kwargs):
        """Load a Whisper model from an MLX-community .npz weights file."""

        config = WhisperConfig.from_pretrained(pretrained_model_name)
        model = WhisperForConditionalGeneration(config)

        npz_path = hf_hub_download(pretrained_model_name, "weights.npz")
        mlx_weights = np.load(npz_path)

        state_dict = model.state_dict()
        for key in state_dict:
            mlx_key = key
            if mlx_key in mlx_weights:
                tensor = torch.from_numpy(mlx_weights[mlx_key].copy())
                if tensor.shape != state_dict[key].shape:
                    tensor = tensor.reshape(state_dict[key].shape)
                state_dict[key] = tensor

        model.load_state_dict(state_dict)

        dtype_override = model_kwargs.get("torch_dtype")
        if dtype_override is not None:
            model.to(dtype_override)

        return model

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
