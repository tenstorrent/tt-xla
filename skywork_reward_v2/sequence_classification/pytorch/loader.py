# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Skywork Reward V2 model loader implementation for sequence classification.
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
    """Available Skywork Reward V2 model variants for sequence classification."""

    LLAMA_3_2_1B = "Llama_3.2_1B"


class ModelLoader(ForgeModel):
    """Skywork Reward V2 model loader implementation for reward scoring.

    This model uses LlamaForSequenceClassification with num_labels=1 to produce
    a single scalar reward score for evaluating conversational responses.
    """

    _VARIANTS = {
        ModelVariant.LLAMA_3_2_1B: LLMModelConfig(
            pretrained_model_name="Skywork/Skywork-Reward-V2-Llama-3.2-1B",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LLAMA_3_2_1B

    # Sample conversation for reward scoring
    sample_text = [
        {"role": "user", "content": "How many people live in Berlin?"},
        {
            "role": "assistant",
            "content": "Berlin had a population of 3,520,031 registered inhabitants.",
        },
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Skywork-Reward-V2",
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

        model_kwargs = {"num_labels": 1}
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

        conv_formatted = self.tokenizer.apply_chat_template(
            self.sample_text, tokenize=False
        )
        inputs = self.tokenizer(
            conv_formatted,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs, inputs=None):
        logits = outputs[0]
        reward_score = logits[0][0].item()
        return f"Reward score: {reward_score:.4f}"
