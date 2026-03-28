# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TRL Internal Testing tiny-GPTNeoXForSequenceClassification model loader implementation.
"""
from transformers import AutoTokenizer, GPTNeoXForSequenceClassification

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ....base import ForgeModel


class ModelLoader(ForgeModel):
    """TRL Internal Testing tiny-GPTNeoXForSequenceClassification model loader."""

    _VARIANTS = {
        "base": ModelConfig(
            pretrained_model_name="trl-internal-testing/tiny-GPTNeoXForSequenceClassification",
        ),
    }

    DEFAULT_VARIANT = "base"

    sample_text = "the movie was great!"

    def __init__(self, variant=None):
        super().__init__(variant)
        self.tokenizer = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant=None) -> ModelInfo:
        if variant is None:
            variant = "base"
        return ModelInfo(
            model="TRL Internal Testing",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        pretrained_model_name = self._variant_config.pretrained_model_name

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = GPTNeoXForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            padding="max_length",
            max_length=32,
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        predicted_value = co_out[0].argmax(-1).item()
        print(f"Predicted Class: {predicted_value}")
