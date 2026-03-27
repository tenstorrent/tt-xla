# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TAPAS model loader implementation for table question answering.
"""

from typing import Optional

import pandas as pd

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


class ModelVariant(StrEnum):
    """Available TAPAS model variants."""

    LARGE_FINETUNED_SQA = "Large_Finetuned_SQA"


class ModelLoader(ForgeModel):
    """TAPAS model loader implementation for table question answering."""

    _VARIANTS = {
        ModelVariant.LARGE_FINETUNED_SQA: LLMModelConfig(
            pretrained_model_name="google/tapas-large-finetuned-sqa",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE_FINETUNED_SQA

    sample_table = pd.DataFrame.from_dict(
        {
            "Repository": ["Transformers", "Datasets", "Tokenizers"],
            "Stars": ["36542", "4512", "3934"],
            "Contributors": ["651", "77", "34"],
            "Programming language": ["Python", "Python", "Rust, Python and NodeJS"],
        }
    )

    sample_query = "How many stars does the transformers repository have?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TAPAS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import TapasTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = TapasTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import TapasForQuestionAnswering

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = TapasForQuestionAnswering.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            table=self.sample_table,
            queries=self.sample_query,
            padding="max_length",
            return_tensors="pt",
        )

        return inputs
