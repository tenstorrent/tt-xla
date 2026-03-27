# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS-Tokenizer model loader implementation for audio feature extraction.
"""
import torch
import torch.nn as nn
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


class Qwen3TTSTokenizerEncoderWrapper(nn.Module):
    """Wrapper around the Qwen3-TTS-Tokenizer encoder.

    Exposes the encoder forward pass that takes a mel spectrogram
    and produces discrete codec tokens for speech tokenization.
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mel):
        return self.encoder(mel)


class ModelVariant(StrEnum):
    """Available Qwen3-TTS-Tokenizer model variants."""

    QWEN3_TTS_TOKENIZER_12HZ = "12Hz"


class ModelLoader(ForgeModel):
    """Qwen3-TTS-Tokenizer model loader for audio feature extraction."""

    _VARIANTS = {
        ModelVariant.QWEN3_TTS_TOKENIZER_12HZ: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-TTS-Tokenizer-12Hz",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_TTS_TOKENIZER_12HZ

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3-TTS-Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.AUDIO_FE,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from qwen_tts import Qwen3TTSTokenizer

        tokenizer = Qwen3TTSTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        model = Qwen3TTSTokenizerEncoderWrapper(tokenizer.encoder)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Encoder expects mel spectrogram input: (batch, n_mels, time_frames)
        # Using 128 mel bins and 100 time frames as representative input
        mel = torch.randn(1, 128, 100, dtype=dtype)
        return (mel,)
