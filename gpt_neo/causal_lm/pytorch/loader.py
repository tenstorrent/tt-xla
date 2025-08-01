# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
GPT-Neo model loader implementation
"""
import torch
from typing import Optional

from transformers import GPTNeoForCausalLM, GPT2Tokenizer, GenerationConfig
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....base import ForgeModel


class ModelVariant(StrEnum):
    """Available GPT-Neo model variants."""

    GPT_NEO_125M = "gpt_neo_125M"
    GPT_NEO_1_3B = "gpt_neo_1_3B"
    GPT_NEO_2_7B = "gpt_neo_2_7B"


class ModelLoader(ForgeModel):
    """GPT-Neo model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.GPT_NEO_125M: LLMModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-125M",
            max_length=256,
        ),
        ModelVariant.GPT_NEO_1_3B: LLMModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-1.3B",
            max_length=256,
        ),
        ModelVariant.GPT_NEO_2_7B: LLMModelConfig(
            pretrained_model_name="EleutherAI/gpt-neo-2.7B",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT_NEO_125M

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
            model="gpt_neo",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the GPT-Neo model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The GPT-Neo model instance.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name)

        # Set pad token to eos token for GPT-Neo
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = GPTNeoForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the GPT-Neo model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) and generation_config that can be fed to the model.
        """

        prompt = (
            "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
            "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
            "researchers was the fact that the unicorns spoke perfect English."
        )

        if self.tokenizer is None:
            model = self.load_model(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        tokenized_inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        generation_config = GenerationConfig(
            max_length=100, do_sample=True, temperature=0.9
        )
        inputs = {
            "input_ids": tokenized_inputs.input_ids,
            "attention_mask": tokenized_inputs.attention_mask,
            "generation_config": generation_config,
        }

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
