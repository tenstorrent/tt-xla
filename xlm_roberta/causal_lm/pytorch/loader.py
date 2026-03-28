# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
XLM-RoBERTa model loader implementation for causal language modeling.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    """Available XLM-RoBERTa causal LM model variants."""

    TINY = "Tiny"


class ModelLoader(ForgeModel):
    """XLM-RoBERTa model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.TINY: LLMModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-xlm-roberta",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY

    sample_text = "Hello, my dog is cute"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="XLM-RoBERTa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            tokenizer: The loaded tokenizer instance
        """
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the XLM-RoBERTa model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The XLM-RoBERTa model instance for causal language modeling.
        """
        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the XLM-RoBERTa model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(self.sample_text, return_tensors="pt")

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs
