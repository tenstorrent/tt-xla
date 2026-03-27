# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MeloTTS-French model loader implementation for text-to-speech tasks.
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

    def forward(self, x, x_lengths, sid, tones, lang_ids, bert, ja_bert):
        audio = self.model.infer(
            x,
            x_lengths,
            sid,
            tones,
            lang_ids,
            bert,
            ja_bert,
        )[0]
        return audio


class ModelVariant(StrEnum):
    """Available MeloTTS model variants."""

    MELOTTS_FRENCH = "French"


class ModelLoader(ForgeModel):
    """MeloTTS-French model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MELOTTS_FRENCH: ModelConfig(
            pretrained_model_name="myshell-ai/MeloTTS-French",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MELOTTS_FRENCH

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

        self._tts = TTS(language="FR", device="cpu")
        model = MeloTTSWrapper(self._tts.model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        from melo import utils

        text = "Bonjour, ceci est un test."
        device = "cpu"
        language = self._tts.language

        bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(
            text, language, self._tts.hps, device, self._tts.symbol_to_id
        )

        x = phones.unsqueeze(0)
        x_lengths = torch.LongTensor([phones.size(0)])
        sid = torch.LongTensor([list(self._tts.hps.data.spk2id.values())[0]])
        tones = tones.unsqueeze(0)
        lang_ids = lang_ids.unsqueeze(0)
        bert = bert.unsqueeze(0)
        ja_bert = ja_bert.unsqueeze(0)

        return x, x_lengths, sid, tones, lang_ids, bert, ja_bert
