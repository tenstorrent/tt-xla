# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ALBERT model loader implementation for token classification.
"""
import torch
from transformers import AlbertForTokenClassification, AlbertTokenizer
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
    """Available ALBERT model variants."""

    BASE_V1 = "base_v1"
    LARGE_V1 = "large_v1"
    XLARGE_V1 = "xlarge_v1"
    XXLARGE_V1 = "xxlarge_v1"
    BASE_V2 = "base_v2"
    LARGE_V2 = "large_v2"
    XLARGE_V2 = "xlarge_v2"
    XXLARGE_V2 = "xxlarge_v2"


class ModelLoader(ForgeModel):
    """ALBERT model loader implementation for token classification tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE_V1: LLMModelConfig(
            pretrained_model_name="albert-base-v1",
            max_length=128,
        ),
        ModelVariant.LARGE_V1: LLMModelConfig(
            pretrained_model_name="albert-large-v1",
            max_length=128,
        ),
        ModelVariant.XLARGE_V1: LLMModelConfig(
            pretrained_model_name="albert-xlarge-v1",
            max_length=128,
        ),
        ModelVariant.XXLARGE_V1: LLMModelConfig(
            pretrained_model_name="albert-xxlarge-v1",
            max_length=128,
        ),
        ModelVariant.BASE_V2: LLMModelConfig(
            pretrained_model_name="albert-base-v2",
            max_length=128,
        ),
        ModelVariant.LARGE_V2: LLMModelConfig(
            pretrained_model_name="albert-large-v2",
            max_length=128,
        ),
        ModelVariant.XLARGE_V2: LLMModelConfig(
            pretrained_model_name="albert-xlarge-v2",
            max_length=128,
        ),
        ModelVariant.XXLARGE_V2: LLMModelConfig(
            pretrained_model_name="albert-xxlarge-v2",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE_V2

    # Shared configuration parameters
    sample_text = "HuggingFace is a company based in Paris and New York"

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
            task=ModelTask.NLP_TOKEN_CLS,
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

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AlbertTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the ALBERT model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ALBERT model instance for token classification.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AlbertForTokenClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ALBERT model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # Create tokenized inputs for the token classification task
        inputs = self.tokenizer(
            self.sample_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            list: Predicted token classes for valid tokens
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        if inputs is None:
            inputs = self.load_inputs()

        predicted_token_class_ids = outputs[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        return predicted_tokens_classes
