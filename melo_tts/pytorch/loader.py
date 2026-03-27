# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeloTTS model loader implementation for text-to-speech tasks.
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
    """Wrapper around MeloTTS SynthesizerTrn to expose infer as forward."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert):
        audio, *_ = self.model.infer(x, x_lengths, sid, tone, language, bert, ja_bert)
        return audio


class ModelVariant(StrEnum):
    """Available MeloTTS model variants."""

    KOREAN = "Korean"


class ModelLoader(ForgeModel):
    """MeloTTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KOREAN: ModelConfig(
            pretrained_model_name="myshell-ai/MeloTTS-Korean",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KOREAN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tts = None

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

        self.tts = TTS(language="KR", device="cpu")
        model = MeloTTSWrapper(self.tts.model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        from melo.text import get_text_for_tts_infer

        text = "안녕하세요."
        bert, ja_bert, phones, tones, lang_ids = get_text_for_tts_infer(
            text, self.tts.hps, "cpu", self.tts.language, self.tts.symbol_to_id
        )

        x = phones.unsqueeze(0)
        x_lengths = torch.LongTensor([phones.size(0)])
        sid = torch.LongTensor([0])
        tone = tones.unsqueeze(0)
        language = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)

        return x, x_lengths, sid, tone, language, bert, ja_bert
