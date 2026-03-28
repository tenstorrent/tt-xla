# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VibeVoice model loader implementation for text-to-speech tasks.
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


class VibeVoiceDecoderWrapper(nn.Module):
    """Wrapper around the VibeVoice decoder (language model) backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces language model logits for speech token prediction.
    """

    def __init__(self, model, lm_head):
        super().__init__()
        self.language_model = model.language_model
        self.lm_head = lm_head

    def forward(self, inputs_embeds):
        outputs = self.language_model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    VIBEVOICE_1_5B = "1.5B"


class ModelLoader(ForgeModel):
    """VibeVoice model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.VIBEVOICE_1_5B: ModelConfig(
            pretrained_model_name="vibevoice/VibeVoice-1.5B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIBEVOICE_1_5B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VibeVoice",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice import (
            VibeVoiceForConditionalGeneration,
        )
        from transformers import AutoConfig, AutoModel

        AutoConfig.register("vibevoice", VibeVoiceConfig)
        AutoModel.register(VibeVoiceConfig, VibeVoiceForConditionalGeneration)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model = VibeVoiceDecoderWrapper(full_model.model, full_model.lm_head)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Decoder hidden_size=1536, use a short sequence of embeddings
        inputs_embeds = torch.randn(1, 32, 1536, dtype=dtype)
        return (inputs_embeds,)
