# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlanT5 model loader implementation
"""
import torch


from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """FlanT5 model loader implementation for Seq2SeqLM."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "google/flan-t5-small"
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="flan_t5",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the FlanT5 model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The FlanT5 model instance for Seq2SeqLM.
        """
        # Initialize tokenizer first with default or overridden dtype

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **model_kwargs)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the FlanT5 model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model(
                dtype_override=dtype_override
            )  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer(
            "A step by step recipe to make bolognese pasta:", return_tensors="pt"
        )
        decoder_input_ids = torch.tensor([[self.tokenizer.pad_token_id]])

        # Batch the inputs using repeat_interleave (works for batch_size=1 too)
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        decoder_input_ids = decoder_input_ids.repeat_interleave(batch_size, dim=0)
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs

    def decode_output(self, outputs, dtype_override=None, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Decoded answer text
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model(
                dtype_override=dtype_override
            )  # This will initialize the tokenizer

        if inputs is None:
            inputs = self.load_inputs()

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        token_ids = torch.argmax(logits, dim=-1)
        decoded = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        # Return single string for batch_size=1, list for batch_size>1
        return decoded[0] if len(decoded) == 1 else decoded
