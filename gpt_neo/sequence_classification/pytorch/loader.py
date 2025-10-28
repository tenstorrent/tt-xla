# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-Neo model loader implementation for sequence classification.
"""
import torch
from transformers import GPTNeoForSequenceClassification, GPT2Tokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available GPT-Neo model variants for sequence classification."""

    GPT_NEO_125M = "gpt_neo_125M"
    GPT_NEO_1_3B = "gpt_neo_1_3B"
    GPT_NEO_2_7B = "gpt_neo_2_7B"


class ModelLoader(ForgeModel):
    """GPT-Neo model loader implementation for sequence classification tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GPT_NEO_125M: ModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-125M",
        ),
        ModelVariant.GPT_NEO_1_3B: ModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-1.3B",
        ),
        ModelVariant.GPT_NEO_2_7B: ModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-2.7B",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT_NEO_125M

    # Shared configuration parameters
    sample_text = "the movie was great!"

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
            model="gpt_neo_seq_cls",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_TEXT_CLS,
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
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for GPT-Neo
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the GPT-Neo model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GPT-Neo model instance for sequence classification.
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

        model = GPTNeoForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Set the pad_token_id in the model config to match the tokenizer
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.pad_token_id

        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the GPT-Neo model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the sequence classification task
        inputs = self.tokenizer(self.sample_text, return_tensors="pt")

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass (logits)
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Predicted category label
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs[0]
        predicted_class_id = logits.argmax().item()
        predicted_category = self.model.config.id2label[predicted_class_id]

        return predicted_category
