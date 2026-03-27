# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TAPEX model loader implementation for table question answering.
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
    """Available TAPEX model variants."""

    BASE_FINETUNED_WIKISQL = "Base_Finetuned_WikiSQL"


class ModelLoader(ForgeModel):
    """TAPEX model loader implementation for table question answering."""

    _VARIANTS = {
        ModelVariant.BASE_FINETUNED_WIKISQL: LLMModelConfig(
            pretrained_model_name="microsoft/tapex-base-finetuned-wikisql",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_FINETUNED_WIKISQL

    sample_table = pd.DataFrame.from_dict(
        {
            "year": [1896, 1900, 1904, 2004, 2008, 2012],
            "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"],
        }
    )

    sample_query = "In which year did beijing host the Olympic Games?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="TAPEX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _linearize_table(table: pd.DataFrame) -> str:
        """Linearize a pandas DataFrame into TAPEX table format."""
        header = " | ".join(table.columns)
        rows = []
        for i, row in table.iterrows():
            row_str = " | ".join(str(v) for v in row.values)
            rows.append(f"row {i + 1} : {row_str}")
        return "col : " + header + " " + " ".join(rows)

    def _load_tokenizer(self, dtype_override=None):
        from transformers import BartTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = BartTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import BartForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        input_text = self.sample_query + " " + self._linearize_table(self.sample_table)

        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        )

        return inputs
