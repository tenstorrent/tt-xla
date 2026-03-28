# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SoLU model loader for causal language modeling using TransformerLens.
"""
import torch
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available SoLU model variants."""

    SOLU_2L512W_C4_CODE = "2L512W_C4_Code"


class ModelLoader(ForgeModel):
    """SoLU loader for causal language modeling via TransformerLens."""

    _VARIANTS = {
        ModelVariant.SOLU_2L512W_C4_CODE: LLMModelConfig(
            pretrained_model_name="NeelNanda/SoLU_2L512W_C4_Code",
            max_length=1024,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SOLU_2L512W_C4_CODE

    sample_text = "The capital of France is"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="SoLU",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = HookedTransformer.from_pretrained(model_name, **model_kwargs)

        return model

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "NeelNanda/gpt-neox-tokenizer-digits"
        )
        return self.tokenizer

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        tokens = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            truncation=True,
        )

        return {"input": tokens["input_ids"]}

    def decode_output(self, outputs, inputs=None):
        """Decode model outputs into human-readable text."""
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        generated_ids = logits.argmax(-1)
        return self.tokenizer.decode(generated_ids[0])
