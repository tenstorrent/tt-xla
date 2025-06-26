# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-Neo model loader implementation
"""
import torch


from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GenerationConfig
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """GPT-Neo model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "EleutherAI/gpt-neo-125M"
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
            model="gpt_neo",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the GPT-Neo model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GPT-Neo model instance.
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = GPTNeoForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GPT-Neo model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """

        prompt = (
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
            "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
            "researchers was the fact that the unicorns spoke perfect English."
        )

        if self.tokenizer is None:
            model = self.load_model(dtype_override=dtype_override)

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        generation_config = GenerationConfig(
            max_length=100, do_sample=True, temperature=0.9
        )
        inputs = {"input_ids": input_ids, "generation_config": generation_config}

        # Replicate inputs for batch size
        for key in inputs:
            if key == "generation_config":
                # GenerationConfig is shared across all batch items
                continue
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

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
