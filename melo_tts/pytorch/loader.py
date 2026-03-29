# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeloTTS-English-v3 model loader implementation for text-to-speech tasks.
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
    """Wrapper around SynthesizerTrn to expose infer as forward."""

    def __init__(self, synthesizer):
        super().__init__()
        self.synthesizer = synthesizer

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert):
        audio = self.synthesizer.infer(
            x,
            x_lengths,
            sid,
            tone,
            language,
            bert,
            ja_bert,
            sdp_ratio=0.2,
            noise_scale=0.6,
            noise_scale_w=0.8,
            length_scale=1.0,
        )[0][0, 0]
        return audio


class ModelVariant(StrEnum):
    """Available MeloTTS model variants."""

    ENGLISH_V3 = "English-v3"


class ModelLoader(ForgeModel):
    """MeloTTS-English-v3 model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.ENGLISH_V3: ModelConfig(
            pretrained_model_name="myshell-ai/MeloTTS-English-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ENGLISH_V3

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

        self._tts = TTS(language="EN_NEWEST", device="cpu")
        model = MeloTTSWrapper(self._tts.model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        from melo import utils as melo_utils

        text = "Hello, this is a test of text to speech."
        bert, ja_bert, phones, tones, lang_ids = melo_utils.get_text_for_tts_infer(
            text, self._tts.language, self._tts.hps, "cpu", self._tts.symbol_to_id
        )

        x = phones.unsqueeze(0)
        x_lengths = torch.LongTensor([phones.size(0)])
        sid = torch.LongTensor([self._tts.hps.data.spk2id["EN-Newest"]])
        tone = tones.unsqueeze(0)
        language = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)

        return x, x_lengths, sid, tone, language, bert, ja_bert
