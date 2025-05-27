# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Falcon model loader implementation for question answering
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, FalconForCausalLM

from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Falcon model loader implementation for question answering tasks."""

    # Shared configuration parameters
    model_name = "tiiuae/Falcon3-1B-Base"
    input_text = "Write a function to calculate the factorial of a number"
    max_length = 512

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Falcon model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Falcon model instance for question answering.
        """
        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        cls.tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(
            cls.model_name, return_dict=False, use_cache=False, **model_kwargs
        )
        return model

    @classmethod
    def load_inputs(cls, dtype_override=None):
        """Load and return sample inputs for the Falcon model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = cls.tokenizer.encode(
            cls.input_text,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=cls.max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs

    @classmethod
    def decode_output(cls, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded answer text
        """
        if not hasattr(cls, "tokenizer"):
            cls.load_model()  # This will initialize the tokenizer

        if inputs is None:
            inputs = cls.load_inputs()

        response_start = torch.argmax(outputs.start_logits)
        response_end = torch.argmax(outputs.end_logits) + 1
        response_tokens = inputs.input_ids[0, response_start:response_end]

        return cls.tokenizer.decode(response_tokens)
