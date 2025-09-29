# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
ALBERT model loader implementation for masked language modeling.
"""

from transformers import FlaxAlbertForMaskedLM, AlbertTokenizer
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
    """Available ALBERT model variants."""

    BASE_V2 = "base_v2"
    LARGE_V2 = "large-v2"
    XLARGE_V2 = "xlarge-v2"
    XXLARGE_V2 = "xxlarge-v2"


class ModelLoader(ForgeModel):
    """ALBERT model loader implementation for masked language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_V2: LLMModelConfig(
            pretrained_model_name="albert/albert-base-v2",
        ),
        ModelVariant.LARGE_V2: LLMModelConfig(
            pretrained_model_name="albert/albert-large-v2",
        ),
        ModelVariant.XLARGE_V2: LLMModelConfig(
            pretrained_model_name="albert/albert-xlarge-v2",
        ),
        ModelVariant.XXLARGE_V2: LLMModelConfig(
            pretrained_model_name="albert/albert-xxlarge-v2",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_V2

    # Shared configuration parameters
    sample_text = "The capital of France is [MASK]."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

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
            model="albert_v2",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
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

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AlbertTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the ALBERT model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            model: The loaded model instance
        """

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        # Load the model
        model = FlaxAlbertForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ALBERT model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the masked language modeling task
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
