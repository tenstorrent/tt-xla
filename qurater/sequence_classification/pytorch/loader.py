# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
QuRater model loader implementation for sequence classification.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
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
    """Available QuRater model variants for sequence classification."""

    QURATER_1_3B = "1.3B"


class ModelLoader(ForgeModel):
    """QuRater model loader implementation for sequence classification tasks.

    QuRater predicts data quality scores across 4 criteria:
    writing style, required expertise, facts and trivia, and educational value.
    """

    _VARIANTS = {
        ModelVariant.QURATER_1_3B: LLMModelConfig(
            pretrained_model_name="princeton-nlp/QuRater-1.3B",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.QURATER_1_3B

    sample_text = "The quick brown fox jumps over the lazy dog."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="QuRater",
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

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        logits = outputs[0]
        labels = [
            "writing_style",
            "required_expertise",
            "facts_and_trivia",
            "educational_value",
        ]
        scores = {label: logits[0, i].item() for i, label in enumerate(labels)}
        return scores
