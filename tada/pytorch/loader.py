# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TADA model loader implementation for text-to-speech tasks.
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


class TadaWrapper(nn.Module):
    """Wrapper around TadaForCausalLM that exposes a clean forward pass.

    The underlying model is a Llama-based causal LM with a diffusion
    acoustic head for text-to-speech synthesis. This wrapper exposes
    the standard causal LM forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)


class ModelVariant(StrEnum):
    """Available TADA model variants."""

    TADA_1B = "1B"


class ModelLoader(ForgeModel):
    """TADA model loader implementation for text-to-speech tasks."""

    _VARIANTS = {
        ModelVariant.TADA_1B: ModelConfig(
            pretrained_model_name="HumeAI/tada-1b",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TADA_1B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TADA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from tada.modules.tada import TadaForCausalLM

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = TadaForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return TadaWrapper(model)

    def load_inputs(self, dtype_override=None):
        from transformers import AutoTokenizer

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
            )

        sample_text = "Please call Stella."
        inputs = self._tokenizer(sample_text, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
