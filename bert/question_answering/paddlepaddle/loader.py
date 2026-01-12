# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT PaddlePaddle model loader implementation for question answering.
"""

from typing import Optional, List

import paddle
from paddlenlp.transformers import BertForQuestionAnswering, BertTokenizer

from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available BERT model variants for question answering (Paddle)."""

    BERT_BASE_UNCASED = "bert-base-uncased"
    BERT_BASE_JAPANESE = "cl-tohoku/bert-base-japanese"
    CHINESE_ROBERTA_BASE = "uer/chinese-roberta-base"


class ModelLoader(ForgeModel):
    """BERT Paddle model loader implementation for question answering."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BERT_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="bert-base-uncased",
        ),
        ModelVariant.BERT_BASE_JAPANESE: LLMModelConfig(
            pretrained_model_name="cl-tohoku/bert-base-japanese",
        ),
        ModelVariant.CHINESE_ROBERTA_BASE: LLMModelConfig(
            pretrained_model_name="uer/chinese-roberta-base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BERT_BASE_UNCASED

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer: Optional[BertTokenizer] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="bert-qa",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.PADDLE,
        )

    def _get_sample_question(self) -> str:
        """Return a sample question based on variant language (matches test inputs_map["..."]["question"])."""
        model_name = self._variant_config.pretrained_model_name
        if "japanese" in model_name:
            return ["中国の首都はどこですか？"]
        if "chinese" in model_name or "roberta" in model_name:
            return ["中国的首都是哪里？"]
        return ["What is the capital of China?"]

    def load_model(self, dtype_override=None):
        """Load Paddle BERT model for question answering."""
        model_name = self._variant_config.pretrained_model_name
        # Initialize tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        model = BertForQuestionAnswering.from_pretrained(model_name)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None) -> List[paddle.Tensor]:
        """Prepare sample inputs for BERT question answering (Paddle).

        Matches the test's usage: tokenizer(question, return_* flags) and convert values to tensors.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        question = self._get_sample_question()
        self.encoded = self.tokenizer(
            question,
            return_token_type_ids=True,
            return_position_ids=True,
            return_attention_mask=True,
        )
        inputs = [paddle.to_tensor(value) for value in self.encoded.values()]
        return inputs

    def decode_output(self, outputs=None):
        """Decode the model output for question answering (mirrors the test)."""
        start_logits, end_logits = outputs
        start_index = start_logits.argmax(dim=-1).item()
        end_index = end_logits.argmax(dim=-1).item()
        answer = self.tokenizer.decode(
            self.encoded["input_ids"][0][start_index : end_index + 1]
        )
        return answer
