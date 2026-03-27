# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ThaiNER model loader implementation for token classification (NER).
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
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
    """Available ThaiNER token classification model variants."""

    PYTHAINLP_THAINER_CORPUS_V2_BASE_MODEL = "pythainlp/thainer-corpus-v2-base-model"


class ModelLoader(ForgeModel):
    """ThaiNER model loader implementation for token classification tasks."""

    _VARIANTS = {
        ModelVariant.PYTHAINLP_THAINER_CORPUS_V2_BASE_MODEL: ModelConfig(
            pretrained_model_name="pythainlp/thainer-corpus-v2-base-model",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PYTHAINLP_THAINER_CORPUS_V2_BASE_MODEL

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.sample_text = (
            "ฉันชื่อ นางสาวมะลิวา บุญสระดี อาศัยอยู่ที่อำเภอนางรอง จังหวัดบุรีรัมย์"
        )
        self.max_length = 128

    @classmethod
    def _get_model_info(cls, variant_name=None):
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ThaiNER",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Context: {self.sample_text}")
        print(f"Predicted Labels: {predicted_tokens_classes}")
