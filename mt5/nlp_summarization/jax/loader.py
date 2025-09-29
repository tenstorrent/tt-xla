# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MT5 model loader implementation for NLP summarization."""

from typing import Optional
from transformers.models.mt5.modeling_flax_mt5 import shift_tokens_right

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
    """Available MT5 model variants."""

    BASE = "base"
    LARGE = "large"
    XL = "xl"


class ModelLoader(ForgeModel):
    """MT5 model loader implementation for NLP summarization."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="google/mt5-base",
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="google/mt5-large",
        ),
        ModelVariant.XL: LLMModelConfig(
            pretrained_model_name="google/mt5-xl",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    sample_text = """summarize: Researchers have extensively studied the benefits of having pets, but the evidence is mixed. Some studies suggest that pets can improve mental health, reduce stress, and increase physical activity.
        However, other studies have found that pets can also contribute to allergies and other health problems. The evidence is still inconclusive, but it is clear that pets can have a positive impact on both physical and mental health."""

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
            model="mt5",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.

        Returns:
            tokenizer: The loaded tokenizer
        """
        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the MT5 model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            The loaded model instance.
        """
        from transformers import FlaxMT5ForConditionalGeneration

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxMT5ForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MT5 model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: The loaded inputs
        """
        from transformers import MT5Config

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load sample inputs
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        config = MT5Config.from_pretrained(self._model_name)

        # MT5 needs decoder_input_ids also as input
        decoder_input_ids = shift_tokens_right(
            inputs["input_ids"],
            config.pad_token_id,
            config.decoder_start_token_id,
        )

        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
