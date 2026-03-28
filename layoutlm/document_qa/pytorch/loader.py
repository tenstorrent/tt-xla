# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LayoutLM document question answering model loader implementation (PyTorch).
"""

import torch
from transformers import AutoTokenizer, LayoutLMForQuestionAnswering
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available LayoutLM document QA model variants."""

    IMPIRA_LAYOUTLM_DOCUMENT_QA = "Impira LayoutLM Document QA"


class ModelLoader(ForgeModel):
    """LayoutLM document question answering model loader implementation."""

    _VARIANTS = {
        ModelVariant.IMPIRA_LAYOUTLM_DOCUMENT_QA: ModelConfig(
            pretrained_model_name="impira/layoutlm-document-qa",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IMPIRA_LAYOUTLM_DOCUMENT_QA

    # Sample document words and their bounding boxes (normalized 0-1000)
    words = ["Invoice", "Number:", "12345", "Date:", "2024-01-15"]
    boxes = [
        [100, 50, 200, 80],
        [210, 50, 330, 80],
        [340, 50, 420, 80],
        [100, 100, 180, 130],
        [190, 100, 340, 130],
    ]
    question = "What is the invoice number?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="LayoutLM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_DOC_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = LayoutLMForQuestionAnswering.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer()

        # Tokenize the question
        question_tokens = self.tokenizer.tokenize(self.question)
        # Tokenize each word individually to track bbox alignment
        word_tokens = []
        word_boxes = []
        for word, box in zip(self.words, self.boxes):
            tokens = self.tokenizer.tokenize(word)
            word_tokens.extend(tokens)
            word_boxes.extend([box] * len(tokens))

        # Build input sequence: [CLS] question [SEP] word_tokens [SEP]
        tokens = (
            [self.tokenizer.cls_token]
            + question_tokens
            + [self.tokenizer.sep_token]
            + word_tokens
            + [self.tokenizer.sep_token]
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # Build bbox: [0,0,0,0] for special tokens and question, actual boxes for words
        bbox = (
            [[0, 0, 0, 0]]  # CLS
            + [[0, 0, 0, 0]] * len(question_tokens)  # question
            + [[0, 0, 0, 0]]  # SEP
            + word_boxes  # document words
            + [[0, 0, 0, 0]]  # SEP
        )

        # token_type_ids: 0 for question, 1 for context
        token_type_ids = [0] * (len(question_tokens) + 2) + [1] * (len(word_tokens) + 1)

        attention_mask = [1] * len(input_ids)

        inputs = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long),
            "bbox": torch.tensor([bbox], dtype=torch.long),
        }

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for document question answering."""
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
        print("Predicted answer:", predicted_answer)
