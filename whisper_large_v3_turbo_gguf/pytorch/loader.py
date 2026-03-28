# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Large V3 Turbo GGUF model loader implementation for automatic speech recognition.
"""
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    AutoProcessor,
    WhisperConfig,
)
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
from ...tools.utils import get_file


class ModelVariant(StrEnum):
    """Available Whisper Large V3 Turbo GGUF model variants."""

    WHISPER_LARGE_V3_TURBO_Q4_K = "Large_v3_Turbo_Q4_K"


class ModelLoader(ForgeModel):
    """Whisper Large V3 Turbo GGUF model loader for automatic speech recognition."""

    _VARIANTS = {
        ModelVariant.WHISPER_LARGE_V3_TURBO_Q4_K: ModelConfig(
            pretrained_model_name="xkeyC/whisper-large-v3-turbo-gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WHISPER_LARGE_V3_TURBO_Q4_K

    GGUF_FILE = "model_q4_k.gguf"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Whisper Large V3 Turbo GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the Whisper Large V3 Turbo GGUF model."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = self.GGUF_FILE

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        ).eval()

        self.processor = WhisperProcessor.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )

        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for the Whisper model."""
        if self.model is None or self.processor is None:
            self.load_model(dtype_override=dtype_override)

        config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )

        # Load audio sample
        weights_pth = get_file("test_files/pytorch/whisper/1272-128104-0000.pt")
        sample = torch.load(weights_pth, weights_only=False)
        sample_audio = sample["audio"]["array"]
        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Preprocess audio
        sampling_rate = 16000
        processor_v3 = AutoProcessor.from_pretrained(
            self._variant_config.pretrained_model_name,
            gguf_file=self.GGUF_FILE,
        )
        features = processor_v3.feature_extractor(
            sample_audio,
            sampling_rate=processor_v3.feature_extractor.sampling_rate,
            return_tensors="pt",
            return_token_timestamps=True,
            return_attention_mask=True,
        )
        input_features = features["input_features"].to(device=device, dtype=dtype)
        attention_mask = features.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Build decoder input IDs
        decoder_prompt_ids = self.processor.get_decoder_prompt_ids(
            task="transcribe", language="en", no_timestamps=True
        )
        init_tokens = [self.model.generation_config.decoder_start_token_id]
        if decoder_prompt_ids:
            init_tokens += [tok for _, tok in decoder_prompt_ids]

        decoder_input_ids = torch.tensor([init_tokens], dtype=torch.long, device=device)
        return [input_features, attention_mask, decoder_input_ids]
