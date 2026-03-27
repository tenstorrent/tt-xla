# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MiniLMv2 model loader implementation for masked language modeling.
"""
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Optional

from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available MiniLMv2 model variants for masked language modeling."""

    MINILMV2_L6_H384_DISTILLED_FROM_BERT_LARGE = (
        "nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large"
    )


class ModelLoader(ForgeModel):
    """MiniLMv2 model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.MINILMV2_L6_H384_DISTILLED_FROM_BERT_LARGE: LLMModelConfig(
            pretrained_model_name="nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINILMV2_L6_H384_DISTILLED_FROM_BERT_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="MiniLMv2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            "The capital of France is [MASK].",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        print("The predicted token for the [MASK] is:", predicted_token)
