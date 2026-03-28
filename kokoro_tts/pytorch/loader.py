# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kokoro-82M model loader implementation for text-to-speech tasks.
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


class KokoroWrapper(nn.Module):
    """Wrapper around KModel to expose forward_with_tokens as forward."""

    def __init__(self, kmodel):
        super().__init__()
        self.kmodel = kmodel

    def forward(self, input_ids, ref_s):
        audio, pred_dur = self.kmodel.forward_with_tokens(input_ids, ref_s)
        return audio


class ModelVariant(StrEnum):
    """Available Kokoro model variants."""

    KOKORO_82M = "82M"
    KOKORO_82M_V1_1_ZH = "82M-v1.1-zh"


class ModelLoader(ForgeModel):
    """Kokoro-82M model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.KOKORO_82M: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M",
        ),
        ModelVariant.KOKORO_82M_V1_1_ZH: ModelConfig(
            pretrained_model_name="hexgrad/Kokoro-82M-v1.1-zh",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.KOKORO_82M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Kokoro",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from kokoro import KPipeline

        lang_code = "z" if self._variant == ModelVariant.KOKORO_82M_V1_1_ZH else "a"
        self.pipeline = KPipeline(
            lang_code=lang_code,
            repo_id=self._variant_config.pretrained_model_name,
            device="cpu",
        )
        model = KokoroWrapper(self.pipeline.model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self._variant == ModelVariant.KOKORO_82M_V1_1_ZH:
            ps = "nǐhǎo ʃìjiè"
            voice_name = "zf_001"
        else:
            ps = "hɛloʊ ðɪs ɪz ɐ tˈɛst"
            voice_name = "af_heart"
        vocab = self.pipeline.model.vocab
        input_ids = [v for p in ps if (v := vocab.get(p)) is not None]
        input_ids = torch.LongTensor([[0, *input_ids, 0]])
        voice = self.pipeline.load_voice(voice_name)
        ref_s = voice[len(ps) - 1]
        return input_ids, ref_s
