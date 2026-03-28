# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BETO Contextualized Hate Speech model loader implementation for sequence classification.
"""

from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available BETO Contextualized Hate Speech model variants for sequence classification."""

    BETO_CONTEXTUALIZED_HATE_SPEECH = "Beto_Contextualized_Hate_Speech"


class ModelLoader(ForgeModel):
    """BETO Contextualized Hate Speech model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.BETO_CONTEXTUALIZED_HATE_SPEECH: ModelConfig(
            pretrained_model_name="piuba-bigdata/beto-contextualized-hate-speech",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BETO_CONTEXTUALIZED_HATE_SPEECH

    sample_text = "Hay que matarlos a todos!!!"
    sample_context = "Debate sobre políticas migratorias en Europa"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"

        return ModelInfo(
            model="BETO_Contextualized_Hate_Speech",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
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

        model = AutoModelForSequenceClassification.from_pretrained(
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
            self.sample_context,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        predicted_value = co_out[0].argmax(-1).item()
        label = self.model.config.id2label[predicted_value]
        print(f"Predicted Label: {label}")
        return label
