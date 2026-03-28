# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PyLingual model loader implementation for token classification (bytecode segmentation).
"""

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
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
    """Available PyLingual token classification model variants."""

    PY313_V1_3_SEGMENTER = "py313_v1.3_segmenter"


class ModelLoader(ForgeModel):
    """PyLingual model loader for token classification (bytecode segmentation)."""

    _VARIANTS = {
        ModelVariant.PY313_V1_3_SEGMENTER: LLMModelConfig(
            pretrained_model_name="syssec-utd/py313-pylingual-v1.3-segmenter",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PY313_V1_3_SEGMENTER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None
        self.sample_text = "LOAD_CONST 0 STORE_NAME 0 LOAD_CONST 1 RETURN_VALUE"
        self.max_length = self._variant_config.max_length

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PyLingual",
            variant=variant,
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
