# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Reference: https://github.com/tenstorrent/tt-buda-demos/blob/main/model_demos/cv_demos/linear_autoencoder/pytorch_linear_autoencoder.py
"""
Codegen model loader implementation
"""


from ...base import ForgeModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelLoader(ForgeModel):
    """Codegen model loader implementation."""

    # Shared configuration parameters
    model_name = "Salesforce/codegen-350M-mono"

    @classmethod
    def load_model(cls, dtype_override=None):
        """Load and return the Codegen model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Codegen model instance.
        """
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForCausalLM.from_pretrained(cls.model_name, **model_kwargs)

        return model

    @classmethod
    def load_inputs(cls, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Codegen model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors, pixel values and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model(
                dtype_override=dtype_override
            )  # This will initialize the tokenizer

        text = "def hello_world():"
        inputs = cls.tokenizer(text, return_tensors="pt")

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    @classmethod
    def decode_outputs(cls, outputs):
        """Decode the model outputs to text.

        Args:
            outputs: The model outputs to decode.

        Returns:
            str: The decoded text.
        """
        # Ensure tokenizer is initialized
        if not hasattr(cls, "tokenizer"):
            cls.load_model()

        # Handle both structured outputs and raw tensors
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Ensure logits are float type for softmax operation
        if not logits.dtype.is_floating_point:
            logits = logits.float()

        # Get logits for the last token in each batch
        next_token_logits = logits[:, -1]
        next_tokens = next_token_logits.softmax(dim=-1).argmax(dim=-1)

        if next_tokens.dim() == 0:
            # Single token case
            return [cls.tokenizer.decode([next_tokens.item()])]
        else:
            # Batch of tokens case
            return [cls.tokenizer.decode([token.item()]) for token in next_tokens]
