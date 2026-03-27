# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Neuphonic NeuTTS Nano model loader implementation for text-to-speech tasks.
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


class NeuTTSWrapper(nn.Module):
    """Wrapper around the NeuTTS Nano LlamaForCausalLM backbone.

    Exposes a clean forward pass that takes input token IDs
    and produces speech token logits.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        return outputs.logits


class ModelVariant(StrEnum):
    """Available NeuTTS Nano model variants."""

    NEUTTS_NANO = "nano"


class ModelLoader(ForgeModel):
    """Neuphonic NeuTTS Nano model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.NEUTTS_NANO: ModelConfig(
            pretrained_model_name="neuphonic/neutts-nano",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.NEUTTS_NANO

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="NeuTTS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype_override or torch.float32,
        )
        model = NeuTTSWrapper(backbone)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        # Short dummy token sequence for the LlamaForCausalLM backbone
        input_ids = torch.randint(0, 1000, (1, 32))
        return (input_ids,)
