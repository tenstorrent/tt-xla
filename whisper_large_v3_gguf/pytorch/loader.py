# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Whisper Large v3 GGUF model loader implementation for automatic speech recognition.

Repository:
- https://huggingface.co/vonjack/whisper-large-v3-gguf
"""
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
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


class ModelVariant(StrEnum):
    """Available Whisper Large v3 GGUF model variants."""

    F16 = "F16"
    Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """Whisper Large v3 GGUF model loader implementation for automatic speech recognition tasks."""

    _VARIANTS = {
        ModelVariant.F16: ModelConfig(
            pretrained_model_name="vonjack/whisper-large-v3-gguf",
        ),
        ModelVariant.Q8_0: ModelConfig(
            pretrained_model_name="vonjack/whisper-large-v3-gguf",
        ),
    }

    _GGUF_FILES = {
        ModelVariant.F16: "whisper-large-v3-f16.gguf",
        ModelVariant.Q8_0: "whisper-large-v3-q8_0.gguf",
    }

    DEFAULT_VARIANT = ModelVariant.Q8_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.processor = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Whisper Large v3 GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        gguf_file = self._GGUF_FILES[self._variant]

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs
        model_kwargs["gguf_file"] = gguf_file

        # Pre-load config from the base Whisper model so that random-weights
        # mode does not need to download/parse the GGUF file just for config.
        config = WhisperConfig.from_pretrained("openai/whisper-large-v3")
        model_kwargs["config"] = config

        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

        model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        ).eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained("openai/whisper-large-v3")

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio waveform (1 second at 16kHz)
        sampling_rate = 16000
        sample_audio = torch.randn(sampling_rate).numpy()

        processor = self.processor(
            sample_audio, return_tensors="pt", sampling_rate=sampling_rate
        )
        input_features = processor.input_features.to(device=device, dtype=dtype)

        decoder_input_ids = torch.full(
            (1, 2),
            model_config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        return [input_features, decoder_input_ids]
