# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi 4 model loader implementation for token classification
"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
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


class ModelVariant(StrEnum):
    """Available Phi 4 model variants."""

    PHI_4 = "microsoft/phi-4"


class ModelLoader(ForgeModel):
    """Phi 4 model loader implementation for token classification tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.PHI_4: ModelConfig(
            pretrained_model_name="microsoft/phi-4",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.PHI_4

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
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="phi-4",
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
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Phi 4 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The Phi 4 model instance for token classification.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForTokenClassification.from_pretrained(
            pretrained_model_name, use_cache=False, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Phi 4 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            List: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Input prompt
        input_prompt = "HuggingFace is a company based in Paris and New York"

        inputs = self.tokenizer(input_prompt, return_tensors="pt")

        # Return as list of tensors as expected by the test
        sample_inputs = [inputs["input_ids"]]

        # Add batch dimension if needed
        if batch_size > 1:
            for i in range(len(sample_inputs)):
                sample_inputs[i] = sample_inputs[i].repeat_interleave(batch_size, dim=0)

        return sample_inputs

    def decode_output(self, outputs, labels=None):
        """Helper method to decode model outputs into human-readable labels.

        Args:
            outputs: Model output from a forward pass (logits)
            labels: Optional list of label names for decoding

        Returns:
            List: Predicted labels for each token
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Get logits from outputs
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        # Get predicted labels
        predictions = torch.argmax(logits, dim=-1)

        # Convert to list
        predicted_labels = predictions.squeeze().tolist()

        # If labels are provided, map indices to label names
        if labels is not None:
            predicted_labels = [labels[pred] for pred in predicted_labels]

        return predicted_labels
