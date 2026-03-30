# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenAssistant OASST-RM Pythia 1.4B reward model loader implementation.
"""
from typing import Optional

from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available OASST-RM reward model variants."""

    PYTHIA_1_4B = "Pythia_1.4B"


class ModelLoader(ForgeModel):
    """OpenAssistant OASST-RM reward model loader implementation."""

    _VARIANTS = {
        ModelVariant.PYTHIA_1_4B: ModelConfig(
            pretrained_model_name="OpenAssistant/oasst-rm-2.1-pythia-1.4b-epoch-2.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PYTHIA_1_4B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OASST-RM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        input_text = (
            "<|prompter|>Hi how are you?<|endoftext|>"
            "<|assistant|>Hi, I am Open-Assistant a large open-source language model "
            "trained by LAION AI. How can I help you today?<|endoftext|>"
        )

        inputs = self.tokenizer(
            input_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        logits = co_out[0]
        reward_score = logits.item()
        print(f"Reward score: {reward_score:.4f}")
