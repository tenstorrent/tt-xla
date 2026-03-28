# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Vietnamese Correction model loader implementation for text2text generation.
"""
import torch
from typing import Optional

from transformers import AutoTokenizer, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import shift_tokens_right

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
    """Available Vietnamese Correction model variants."""

    VIETNAMESE_CORRECTION = "Vietnamese_Correction"


class ModelLoader(ForgeModel):
    """Vietnamese Correction model loader implementation for text2text generation."""

    _VARIANTS = {
        ModelVariant.VIETNAMESE_CORRECTION: LLMModelConfig(
            pretrained_model_name="bmd1905/vietnamese-correction",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.VIETNAMESE_CORRECTION

    sample_text = (
        "côn viec kin doanh thì rất kho khan nên toi quyết dinh chuyển sang nghề khac"
    )

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
            model="Vietnamese-Correction",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
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

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Vietnamese Correction model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BARTpho model instance for text correction.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Vietnamese Correction model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            truncation=True,
            return_tensors="pt",
        )

        # Seq2seq models need decoder_input_ids for the forward pass.
        model = self.load_model(dtype_override=dtype_override)
        decoder_input_ids = shift_tokens_right(
            inputs["input_ids"],
            model.config.pad_token_id,
            model.config.decoder_start_token_id,
        )

        inputs["decoder_input_ids"] = decoder_input_ids

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
