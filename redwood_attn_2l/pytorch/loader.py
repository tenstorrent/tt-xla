# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ArthurConmy/redwood_attn_2l model loader implementation for causal language modeling.

This is a 2-layer attention-only transformer from the Redwood Research
mechanistic interpretability project. It uses a GPT-2-style architecture with
no MLP layers and is loaded via the TransformerLens library.
"""

from transformer_lens import HookedTransformer
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


class ModelVariant(StrEnum):
    """Available Redwood Attn 2L model variants."""

    REDWOOD_ATTN_2L = "Default"


class ModelLoader(ForgeModel):
    """Redwood Attn 2L model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.REDWOOD_ATTN_2L: ModelConfig(
            pretrained_model_name="redwood_attn_2l",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.REDWOOD_ATTN_2L

    sample_text = "The quick brown fox jumps over the lazy dog"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Redwood Attn 2L",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model = HookedTransformer.from_pretrained(model_name, **kwargs)
        self.tokenizer = model.tokenizer

        model.eval()

        if dtype_override is not None:
            model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        tokens = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=256,
        )

        return {"input": tokens["input_ids"]}
