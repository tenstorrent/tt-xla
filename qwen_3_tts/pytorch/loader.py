# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS model loader implementation for text-to-speech tasks.
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


class Qwen3TTSTalkerWrapper(nn.Module):
    """Wrapper around the Qwen3-TTS talker backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    (prefill mode) and produces codec logits for speech synthesis.
    """

    def __init__(self, talker):
        super().__init__()
        self.model = talker.model
        self.codec_head = talker.codec_head

    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = self.codec_head(outputs.last_hidden_state)
        return logits


class ModelVariant(StrEnum):
    """Available Qwen3-TTS model variants."""

    QWEN3_TTS_1_7B_VOICE_DESIGN = "1.7B-VoiceDesign"


class ModelLoader(ForgeModel):
    """Qwen3-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_TTS_1_7B_VOICE_DESIGN: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_TTS_1_7B_VOICE_DESIGN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3-TTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from qwen_tts.core.models import (
            Qwen3TTSConfig,
            Qwen3TTSForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel

        AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
        AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            dtype=dtype_override or torch.float32,
        )
        model = Qwen3TTSTalkerWrapper(full_model.talker)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Talker hidden_size=2048, use a short sequence of embeddings
        inputs_embeds = torch.randn(1, 32, 2048, dtype=dtype)
        return (inputs_embeds,)
