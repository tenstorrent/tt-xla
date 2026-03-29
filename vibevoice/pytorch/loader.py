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
    """Wrapper around the VibeVoice decoder (Qwen2-based LLM).

    Exposes a clean forward pass that takes pre-computed input embeddings
    and produces hidden states from the decoder backbone.
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, inputs_embeds):
        outputs = self.decoder(inputs_embeds=inputs_embeds, use_cache=False)
        return outputs.last_hidden_state


class ModelVariant(StrEnum):
    """Available VibeVoice model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """VibeVoice model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.LARGE: ModelConfig(
            pretrained_model_name="aoi-ot/VibeVoice-Large",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

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
        """Load the VibeVoice model and wrap the decoder component."""
        from transformers import AutoModel

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        full_model = AutoModel.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            **model_kwargs,
        )
        model = VibeVoiceDecoderWrapper(full_model.decoder)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Generate synthetic input embeddings for the VibeVoice decoder."""
        dtype = dtype_override or torch.float32
        # Decoder hidden_size=3584, use a short sequence of embeddings
        inputs_embeds = torch.randn(1, 32, 3584, dtype=dtype)
        return (inputs_embeds,)
