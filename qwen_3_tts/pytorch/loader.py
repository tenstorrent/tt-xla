# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 TTS model loader implementation for text-to-speech tasks.
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
    """Wrapper around the Qwen3 TTS talker backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces codec logits via the talker transformer and codec head.
    """

    def __init__(self, talker):
        super().__init__()
        self.talker = talker

    def forward(self, inputs_embeds):
        outputs = self.talker.model.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=False,
            use_cache=False,
        )
        hidden_states = outputs.last_hidden_state
        codec_logits = self.talker.codec_head(hidden_states)
        return codec_logits


class ModelVariant(StrEnum):
    """Available Qwen3 TTS model variants."""

    QWEN3_TTS_12HZ_CUSTOM_VOICE = "12Hz-1.7B-CustomVoice"


class ModelLoader(ForgeModel):
    """Qwen3 TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.QWEN3_TTS_12HZ_CUSTOM_VOICE: ModelConfig(
            pretrained_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QWEN3_TTS_12HZ_CUSTOM_VOICE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Qwen3TTS",
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

        config = Qwen3TTSConfig.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        full_model = Qwen3TTSForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name,
            config=config,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        model = Qwen3TTSTalkerWrapper(full_model.talker)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Talker hidden_size=1024, use a short sequence
        inputs_embeds = torch.randn(1, 32, 1024, dtype=dtype)
        return (inputs_embeds,)
