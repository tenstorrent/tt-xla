# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BART Yahoo Answers model loader implementation for Zero-Shot Sequence Classification.
"""
from transformers import BartForSequenceClassification, BartTokenizer
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
from transformers.models.bart.modeling_bart import shift_tokens_right


class ModelVariant(StrEnum):
    """Available BART Yahoo Answers model variants."""

    LARGE = "Large"


class ModelLoader(ForgeModel):
    """BART Yahoo Answers model loader for zero-shot sequence classification."""

    _VARIANTS = {
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="joeddav/bart-large-mnli-yahoo-answers",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    premise = "Who are you voting for in 2020?"
    hypothesis = "This text is about politics."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BART Yahoo Answers",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = BartTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            pad_to_max_length=True,
            **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs_dict = self.tokenizer(
            self.premise,
            self.hypothesis,
            truncation="only_first",
            padding="max_length",
            max_length=256,
            return_tensors="pt",
        )

        model = self.load_model()
        decoder_input_ids = shift_tokens_right(
            inputs_dict["input_ids"],
            model.config.pad_token_id,
            model.config.decoder_start_token_id,
        )
        inputs = [
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            decoder_input_ids,
        ]
        return inputs
