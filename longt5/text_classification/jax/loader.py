# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LongT5 model loader implementation for Conditional Generation.
"""

from typing import Optional
from transformers.models.longt5.modeling_flax_longt5 import shift_tokens_right

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
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available LongT5 model variants."""

    BASE_TGLOBAL = "base-tglobal"
    LARGE_LOCAL = "large-local"
    XL_TGLOBAL = "xl-tglobal"


class ModelLoader(ForgeModel):
    """LongT5 model loader implementation for text classification."""

    _VARIANTS = {
        ModelVariant.BASE_TGLOBAL: LLMModelConfig(
            pretrained_model_name="google/long-t5-tglobal-base",
        ),
        ModelVariant.LARGE_LOCAL: LLMModelConfig(
            pretrained_model_name="google/long-t5-local-large",
        ),
        ModelVariant.XL_TGLOBAL: LLMModelConfig(
            pretrained_model_name="google/long-t5-tglobal-xl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_TGLOBAL

    sample_text = "summarize: My friends are cool but they eat too many carbs."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

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
            model="longt5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer instance
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the LongT5 model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        """

        from transformers import FlaxLongT5ForConditionalGeneration

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load the model
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model = FlaxLongT5ForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the LongT5 model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: The loaded inputs
        """
        from transformers import LongT5Config

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load sample inputs
        inputs_dict = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        config = LongT5Config.from_pretrained(self._model_name)

        # LongT5 needs decoder_input_ids also as input
        decoder_input_ids = shift_tokens_right(
            inputs_dict["input_ids"],
            config.pad_token_id,
            config.decoder_start_token_id,
        )

        inputs = {
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
        }

        return inputs
