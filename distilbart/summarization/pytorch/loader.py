# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBART model loader implementation for summarization.
"""

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
    """Available DistilBART model variants for summarization."""

    CNN_6_6 = "CNN_6_6"


class ModelLoader(ForgeModel):
    """DistilBART model loader implementation for summarization."""

    _VARIANTS = {
        ModelVariant.CNN_6_6: LLMModelConfig(
            pretrained_model_name="sshleifer/distilbart-cnn-6-6",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CNN_6_6

    sample_text = (
        "The tower is 324 metres (1,063 ft) tall, about the same height as an "
        "81-storey building, and the tallest structure in Paris. Its base is square, "
        "measuring 125 metres (410 ft) on each side. It was the first structure to "
        "reach a height of 300 metres. Excluding transmitters, the Eiffel Tower is "
        "the second tallest free-standing structure in France after the Millau Viaduct."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="DistilBART",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
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

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
            truncation=True,
        )

        return inputs
