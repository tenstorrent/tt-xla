# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Oxide Lab Whisper Large V3 GGUF model loader implementation.

oxide-lab/whisper-large-v3-GGUF is a GGUF-quantized version of OpenAI's
Whisper Large V3 model for automatic speech recognition.
"""

import numpy as np
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
    """Available Oxide Lab Whisper GGUF model variants."""

    LARGE_V3_Q4_0 = "large-v3-q4_0"


class ModelLoader(ForgeModel):
    """Oxide Lab Whisper Large V3 GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.LARGE_V3_Q4_0: ModelConfig(
            pretrained_model_name="oxide-lab/whisper-large-v3-GGUF",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_V3_Q4_0

    GGUF_FILE = "whisper-large-v3-q4_0.gguf"

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Oxide_Lab_Whisper_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_ASR,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        super().__init__(variant)
        self.processor = None
        self.model = None

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load an Oxide Lab Whisper GGUF model from Hugging Face."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"use_cache": False, "gguf_file": self.GGUF_FILE}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = WhisperForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.processor = WhisperProcessor.from_pretrained(
            pretrained_model_name, gguf_file=self.GGUF_FILE
        )

        self.model.eval()
        if dtype_override is not None:
            self.model.to(dtype_override)
        return self.model

    def load_inputs(self, dtype_override=None):
        """Generate sample inputs for Oxide Lab Whisper GGUF model."""
        if self.model is None or self.processor is None:
            self.load_model()

        model_config = WhisperConfig.from_pretrained(
            self._variant_config.pretrained_model_name, gguf_file=self.GGUF_FILE
        )

        model_param = next(self.model.parameters())
        device, dtype = model_param.device, dtype_override or model_param.dtype

        # Generate synthetic audio and process through the feature extractor
        sample_audio = np.random.randn(16000 * 3).astype(np.float32)
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
