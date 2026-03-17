# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XLMRoberta For Masked LM model loader implementation
"""
import torch
from transformers import AutoTokenizer, XLMRobertaForMaskedLM
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
    """Available XLMRoberta For Masked LM model variants."""

    XLM_BASE = "Xlm_Base"


class ModelLoader(ForgeModel):
    """XLMRoberta For Masked LM model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.XLM_BASE: ModelConfig(
            pretrained_model_name="FacebookAI/xlm-roberta-base",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.XLM_BASE

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
            model="RoBERTa",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the XLMRoberta For Masked LM model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The XLMRoberta For Masked LM model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load pre-trained model from HuggingFace
        model = XLMRobertaForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the XLMRoberta For Masked LM model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer()

        # Sample text input
        test_input = "The capital of France is <mask>."

        # Tokenize input
        inputs = self.tokenizer(test_input, return_tensors="pt")

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        """Helper method to decode model outputs for masked language modeling.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded predicted token for the mask position
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        # Handle data parallel case (list of tensors)
        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            # Get logits from outputs
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Load inputs to find mask position
        inputs = self.load_inputs()

        # Retrieve index of <mask> token
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]

        # Get predicted token ID for mask position
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

        # Decode the predicted token
        output = self.tokenizer.decode(predicted_token_id)

        return f"Output: {output}"
