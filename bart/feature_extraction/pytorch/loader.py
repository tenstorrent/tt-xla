# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BART model loader implementation for feature extraction.
"""
from transformers import AutoTokenizer, BartModel
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
from transformers.models.bart.modeling_bart import shift_tokens_right


class ModelVariant(StrEnum):
    """Available BART model variants for feature extraction."""

    TINY = "Tiny"


class ModelLoader(ForgeModel):
    """BART model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.TINY: LLMModelConfig(
            pretrained_model_name="trl-internal-testing/tiny-BartModel",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY

    sample_text = "Hello there fellow traveller, how are you doing today?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None

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
            model="BART",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
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
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BART model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BART model instance for feature extraction.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartModel.from_pretrained(pretrained_model_name, **model_kwargs)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BART model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors that can be fed to the model.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs_dict = self._tokenizer(
            self.sample_text,
            truncation=True,
            padding="max_length",
            max_length=self._variant_config.max_length,
            return_tensors="pt",
        )

        model = self.load_model()
        decoder_input_ids = shift_tokens_right(
            inputs_dict["input_ids"],
            model.config.pad_token_id,
            model.config.decoder_start_token_id,
        )

        inputs = [
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            decoder_input_ids,
        ]

        return inputs
