# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MOSS-TTS model loader implementation for text-to-speech tasks.
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


class MossTTSWrapper(nn.Module):
    """Wrapper around the MOSS-TTS backbone.

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces logits for speech token prediction.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model.model
        self.lm_head = model.lm_head

    def forward(self, inputs_embeds):
        outputs = self.model(inputs_embeds=inputs_embeds, use_cache=False)
        logits = self.lm_head(outputs.last_hidden_state)
        return logits


class ModelVariant(StrEnum):
    """Available MOSS-TTS model variants."""

    MOSS_TTS_DELAY_8B = "Delay-8B"


class ModelLoader(ForgeModel):
    """MOSS-TTS model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.MOSS_TTS_DELAY_8B: ModelConfig(
            pretrained_model_name="OpenMOSS-Team/MOSS-TTS",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSS_TTS_DELAY_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MOSS-TTS",
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
            **kwargs,
        )
        model = MossTTSWrapper(full_model)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        dtype = dtype_override or torch.float32
        # Use a short sequence of embeddings matching the model's hidden size
        inputs_embeds = torch.randn(1, 32, 4096, dtype=dtype)
        return (inputs_embeds,)
