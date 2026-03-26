# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chatterbox TTS model loader implementation for text-to-speech tasks.
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


class ChatterboxT3Wrapper(nn.Module):
    """Wrapper around the Chatterbox T3 Llama backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces speech token logits.
    """

    def __init__(self, t3):
        super().__init__()
        self.t3 = t3

    def forward(self, inputs_embeds):
        tfmr_out = self.t3.tfmr.forward(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden_states = tfmr_out.hidden_states[-1]
        speech_logits = self.t3.speech_head(hidden_states)
        return speech_logits


class ModelVariant(StrEnum):
    """Available Chatterbox model variants."""

    CHATTERBOX = "chatterbox"


class ModelLoader(ForgeModel):
    """Chatterbox TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.CHATTERBOX: ModelConfig(
            pretrained_model_name="ResembleAI/chatterbox",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CHATTERBOX

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Chatterbox",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from chatterbox.tts import ChatterboxTTS

        tts = ChatterboxTTS.from_pretrained(device="cpu")
        model = ChatterboxT3Wrapper(tts.t3)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # T3 Llama backbone hidden_size=1024, use a short sequence
        inputs_embeds = torch.randn(1, 32, 1024, dtype=dtype)
        return (inputs_embeds,)
