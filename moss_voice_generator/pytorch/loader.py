# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-VoiceGenerator model loader implementation for text-to-speech generation.
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


class MossVoiceGeneratorWrapper(nn.Module):
    """Wrapper around the MOSS-VoiceGenerator backbone.

    Exposes a clean forward pass through the Qwen3-based language model
    backbone, producing hidden states suitable for audio code prediction.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


class ModelVariant(StrEnum):
    """Available MOSS-VoiceGenerator model variants."""

    MOSS_VOICE_GENERATOR = "moss-voice-generator"


class ModelLoader(ForgeModel):
    """MOSS-VoiceGenerator model loader for text-to-speech generation."""

    _VARIANTS = {
        ModelVariant.MOSS_VOICE_GENERATOR: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-VoiceGenerator",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSS_VOICE_GENERATOR

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-VoiceGenerator",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModel

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            torch_dtype=dtype_override or torch.float32,
        )
        model = MossVoiceGeneratorWrapper(full_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        # Simulate tokenized input: batch of 1, short sequence
        input_ids = torch.randint(0, 1000, (1, 32))
        attention_mask = torch.ones(1, 32, dtype=torch.long)
        return (input_ids, attention_mask)
