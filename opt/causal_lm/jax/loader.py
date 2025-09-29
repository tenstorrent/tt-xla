# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""OPT model loader implementation for causal language modeling."""

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
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available OPT model variants."""

    _1_3B = "1.3B"
    _2_7B = "2.7B"
    _6_7B = "6.7B"
    _125M = "125M"
    _350M = "350M"


class ModelLoader(ForgeModel):
    """OPT model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant._1_3B: LLMModelConfig(
            pretrained_model_name="facebook/opt-1.3b",
        ),
        ModelVariant._2_7B: LLMModelConfig(
            pretrained_model_name="facebook/opt-2.7b",
        ),
        ModelVariant._6_7B: LLMModelConfig(
            pretrained_model_name="facebook/opt-6.7b",
        ),
        ModelVariant._125M: LLMModelConfig(
            pretrained_model_name="facebook/opt-125m",
        ),
        ModelVariant._350M: LLMModelConfig(
            pretrained_model_name="facebook/opt-350m",
        ),
    }

    DEFAULT_VARIANT = ModelVariant._1_3B

    sample_text = "Hello there fellow traveller"

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
            model="opt",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
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
        """Load and return the OPT model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            The loaded model instance.
        """
        from transformers import FlaxOPTForCausalLM

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxOPTForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the OPT model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: The loaded inputs
        """

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load sample inputs
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
