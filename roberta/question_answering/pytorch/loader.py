# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoBERTa model loader implementation for question answering.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available RoBERTa question answering model variants."""

    DEEPSET_ROBERTA_BASE_SQUAD2 = "deepset_roberta_base_squad2"
    ARMAGEDDON_ROBERTA_LARGE_SQUAD2_COVID_QA_DEEPSET = (
        "armageddon_roberta_large_squad2_covid_qa_deepset"
    )
    UER_ROBERTA_BASE_CHINESE_EXTRACTIVE_QA = "uer_roberta_base_chinese_extractive_qa"


class ModelLoader(ForgeModel):
    """RoBERTa model loader implementation for question answering tasks."""

    _VARIANTS = {
        ModelVariant.DEEPSET_ROBERTA_BASE_SQUAD2: LLMModelConfig(
            pretrained_model_name="deepset/roberta-base-squad2",
            max_length=384,
        ),
        ModelVariant.ARMAGEDDON_ROBERTA_LARGE_SQUAD2_COVID_QA_DEEPSET: LLMModelConfig(
            pretrained_model_name="armageddon/roberta-large-squad2-covid-qa-deepset",
            max_length=384,
        ),
        ModelVariant.UER_ROBERTA_BASE_CHINESE_EXTRACTIVE_QA: LLMModelConfig(
            pretrained_model_name="uer/roberta-base-chinese-extractive-qa",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DEEPSET_ROBERTA_BASE_SQUAD2

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.max_length = self._variant_config.max_length

        self.context = (
            "Super Bowl 50 was an American football game to determine the champion of the National Football League "
            "(NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the "
            "National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. "
            "The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California."
        )
        self.question = "Which NFL team represented the AFC at Super Bowl 50?"

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="RoBERTa",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.question,
            self.context,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        start_logits = co_out[0]
        end_logits = co_out[1]

        answer_start_index = start_logits.argmax()
        answer_end_index = end_logits.argmax()

        input_ids = inputs["input_ids"]
        predict_answer_tokens = input_ids[0, answer_start_index : answer_end_index + 1]

        predicted_answer = self.tokenizer.decode(
            predict_answer_tokens, skip_special_tokens=True
        )
        print(f"Question: {self.question}")
        print(f"Predicted answer: {predicted_answer}")
