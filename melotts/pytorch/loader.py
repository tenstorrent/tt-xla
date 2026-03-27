# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeloTTS Chinese model loader implementation for text-to-speech tasks.
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


class MeloTTSWrapper(nn.Module):
    """Wrapper around the MeloTTS synthesizer for Chinese TTS.

    Exposes the VITS synthesizer's infer method as a clean forward pass
    that takes pre-computed text tokens, tones, language IDs, and BERT
    features to produce audio waveforms.
    """

    def __init__(self, synthesizer):
        super().__init__()
        self.synthesizer = synthesizer

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert):
        audio, attn, mask, (z, z_p, m_p, logs_p) = self.synthesizer.infer(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            ja_bert,
        )
        return audio


class ModelVariant(StrEnum):
    """Available MeloTTS model variants."""

    MELOTTS_CHINESE = "chinese"


class ModelLoader(ForgeModel):
    """MeloTTS Chinese model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MELOTTS_CHINESE: ModelConfig(
            pretrained_model_name="myshell-ai/MeloTTS-Chinese",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MELOTTS_CHINESE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tts = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MeloTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from melo.api import TTS

        self._tts = TTS(language="ZH", device="cpu")
        model = MeloTTSWrapper(self._tts.model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # The VITS synthesizer expects:
        #   x: [batch, seq_len] - text token IDs
        #   x_lengths: [batch] - sequence lengths
        #   sid: [batch] - speaker IDs
        #   tone: [batch, seq_len] - tone IDs (important for Chinese)
        #   language: [batch, seq_len] - language IDs
        #   bert: [batch, bert_dim, seq_len] - Chinese BERT features
        #   ja_bert: [batch, ja_bert_dim, seq_len] - Japanese BERT features (zeros for Chinese)
        seq_len = 32
        bert_dim = 1024
        ja_bert_dim = 768

        x = torch.randint(0, 100, (1, seq_len), dtype=torch.long)
        x_lengths = torch.tensor([seq_len], dtype=torch.long)
        sid = torch.tensor([0], dtype=torch.long)
        tone = torch.randint(0, 5, (1, seq_len), dtype=torch.long)
        language = torch.zeros(1, seq_len, dtype=torch.long)
        bert = torch.randn(1, bert_dim, seq_len, dtype=dtype)
        ja_bert = torch.zeros(1, ja_bert_dim, seq_len, dtype=dtype)

        return x, x_lengths, sid, tone, language, bert, ja_bert
