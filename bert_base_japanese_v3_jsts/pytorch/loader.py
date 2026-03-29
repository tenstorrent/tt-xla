# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT Base Japanese v3 JSTS model loader implementation for semantic textual similarity.
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """BERT Base Japanese v3 JSTS model loader for semantic textual similarity."""

    def __init__(self, variant=None):
        super().__init__(variant)

        self.model_name = "llm-book/bert-base-japanese-v3-jsts"
        self.max_length = 128
        self.tokenizer = None
        self.text = "川べりでサーフボードを持った人たちがいます"
        self.text_pair = "サーファーたちが川べりに立っています"

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="BERT_Base_Japanese_v3_JSTS",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.text,
            self.text_pair,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        score = co_out[0].item()
        print(f"Similarity Score: {score:.4f}")
