# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MT5 model loader implementation for multilingual summarization.
"""

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
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
    """Available MT5 multilingual summarization model variants."""

    MULTILINGUAL_XLSUM = "Multilingual_XLSum"


class ModelLoader(ForgeModel):
    """MT5 model loader implementation for multilingual summarization tasks."""

    _VARIANTS = {
        ModelVariant.MULTILINGUAL_XLSUM: LLMModelConfig(
            pretrained_model_name="csebuetnlp/mT5_multilingual_XLSum",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTILINGUAL_XLSUM

    sample_text = (
        "Videos that say approved vaccines are dangerous and "
        "cause autism, cancer or infertility are among those that "
        "will be removed, the company said. The policy includes "
        "the  eli mi nation of false claims that say the substances "
        "in approved COVID-19 vaccines are harmful."
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

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
            model="mT5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MT5 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MT5 model instance for summarization.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MT5 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
