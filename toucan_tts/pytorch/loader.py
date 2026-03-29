# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ToucanTTS model loader implementation for text-to-speech tasks.

ToucanTTS is a massively multilingual text-to-speech system built on a
modified FastSpeech 2 architecture with Conformer encoder/decoder blocks.
This loader exposes the ToucanTTS synthesizer so that its forward pass
can be compiled and profiled independently of the vocoder.
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


class ToucanTTSWrapper(nn.Module):
    """Wrapper around the ToucanTTS synthesizer for a traceable forward pass.

    The ToucanTTS model (FastSpeech 2 with Conformer blocks) takes
    articulatory phone features and produces mel spectrograms.  This
    wrapper packages the inference call into a clean forward signature.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, text_lengths, utterance_embedding, lang_ids):
        return self.model.inference(
            text=text,
            text_lengths=text_lengths,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_ids,
        )


class ModelVariant(StrEnum):
    """Available ToucanTTS model variants."""

    MULTILINGUAL = "Multilingual"


class ModelLoader(ForgeModel):
    """ToucanTTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MULTILINGUAL: ModelConfig(
            pretrained_model_name="Flux9665/ToucanTTS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTILINGUAL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="ToucanTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from IMSToucan.InferenceInterfaces.ToucanTTSInterface import (
            ToucanTTSInterface,
        )

        self.tts = ToucanTTSInterface(
            device="cpu", tts_model_path="Meta", faster_vocoder=True
        )
        wrapper = ToucanTTSWrapper(self.tts.model)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        batch = 1
        text_len = 32
        phone_feature_dim = 62

        text = torch.randn(batch, text_len, phone_feature_dim, dtype=dtype)
        text_lengths = torch.tensor([text_len], dtype=torch.long)
        utterance_embedding = torch.randn(batch, 64, dtype=dtype)
        lang_ids = torch.tensor([0], dtype=torch.long)

        return text, text_lengths, utterance_embedding, lang_ids
