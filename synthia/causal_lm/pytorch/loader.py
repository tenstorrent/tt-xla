# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Synthia v3.0 11B AWQ model loader implementation for causal language modeling.

AWQ-quantized Solar-based 11B model from TheBloke, fine-tuned for chain-of-thought reasoning.
"""

import torch
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Synthia model variants."""

    SYNTHIA_V3_0_11B_AWQ = "Synthia_v3.0_11B_Awq"


class ModelLoader(ForgeModel):
    """Synthia v3.0 11B AWQ model loader for causal language modeling."""

    _VARIANTS = {
        ModelVariant.SYNTHIA_V3_0_11B_AWQ: LLMModelConfig(
            pretrained_model_name="TheBloke/Synthia-v3.0-11B-AWQ",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SYNTHIA_V3_0_11B_AWQ

    _AWQ_VARIANTS = frozenset({ModelVariant.SYNTHIA_V3_0_11B_AWQ})

    sample_text = "Elaborate on the topic of artificial general intelligence using a Tree of Thoughts approach."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Synthia",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
            **tokenizer_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self._variant in self._AWQ_VARIANTS:
            model_kwargs["device_map"] = "cpu"

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
