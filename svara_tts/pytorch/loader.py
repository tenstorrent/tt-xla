# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Svara TTS voice clone model loader implementation for text-to-speech tasks.
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


class SvaraTTSWrapper(nn.Module):
    """Wrapper around the Svara TTS LlamaForCausalLM backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces next-token logits over the expanded vocabulary (including
    audio tokens).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds):
        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            return_dict=True,
        )
        return outputs.logits


class ModelVariant(StrEnum):
    """Available Svara TTS model variants."""

    VOICECLONE_BETA = "voiceclone-beta"


class ModelLoader(ForgeModel):
    """Svara TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.VOICECLONE_BETA: ModelConfig(
            pretrained_model_name="kenpath/svara-tts-voiceclone-beta",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VOICECLONE_BETA

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="SvaraTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForCausalLM

        full_model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype_override or torch.float32,
        )

        model = SvaraTTSWrapper(full_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Llama 3.2-based backbone with hidden_size=3072, use a short sequence
        inputs_embeds = torch.randn(1, 32, 3072, dtype=dtype)
        return (inputs_embeds,)
