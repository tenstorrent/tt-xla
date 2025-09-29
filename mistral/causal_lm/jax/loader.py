# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Mistral model loader implementation for causal language modeling.
"""

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
    """Available Mistral model variants."""

    V0_1 = "v0_1"
    V0_1_TINY = "v0_1_tiny"
    V0_2_INSTRUCT = "v0_2_instruct"
    V0_3_INSTRUCT = "v0_3_instruct"


class ModelLoader(ForgeModel):
    """Mistral model loader implementation for causal language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.V0_1: LLMModelConfig(
            pretrained_model_name="ksmcg/Mistral-7B-v0.1",
        ),
        ModelVariant.V0_1_TINY: LLMModelConfig(
            pretrained_model_name="ksmcg/Mistral-tiny",
        ),
        ModelVariant.V0_2_INSTRUCT: LLMModelConfig(
            pretrained_model_name="unsloth/mistral-7b-instruct-v0.2",
        ),
        ModelVariant.V0_3_INSTRUCT: LLMModelConfig(
            pretrained_model_name="unsloth/mistral-7b-instruct-v0.3",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.V0_1

    sample_text = "Hello there fellow traveler"

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
            model="mistral",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load the tokenizer for the model.
        Args:
            dtype_override: Optional dtype to override the default dtype.
        Returns:
            Tokenizer: The tokenizer for the model
        """

        from transformers import AutoTokenizer

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def _is_v02_or_later(self) -> bool:
        """Check if the current variant is v0.2 or later (requires sliding window fix)."""
        return self._variant in [ModelVariant.V0_2_INSTRUCT, ModelVariant.V0_3_INSTRUCT]

    def load_model(self, dtype_override=None):
        """Load and return the Mistral model instance for this instance's variant.
        Args:
            dtype_override: Optional dtype to override the default dtype.
        Returns:
            model: The loaded model instance
        """
        from transformers import FlaxMistralForCausalLM, MistralConfig

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # For v0.2 and later models, we need to handle sliding window configuration
        if self._is_v02_or_later():
            # Initialize model with custom config to fix sliding window issue
            # From v0.2 version of Mistral-7B sliding window attention was removed,
            # but Transformers Flax implementation wasn't updated to take that into account
            config = MistralConfig.from_pretrained(pretrained_model_name)
            config.sliding_window = config.max_position_embeddings
            model = FlaxMistralForCausalLM(config, **model_kwargs)
        else:
            # Load the model normally for v0.1 variants
            model = FlaxMistralForCausalLM.from_pretrained(
                pretrained_model_name, **model_kwargs
            )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Mistral model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the causal language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
