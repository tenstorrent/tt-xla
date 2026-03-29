# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Dummy Tokenizer Fast (ALBERT) For Masked LM model loader implementation

This is a dummy ALBERT model with a fast tokenizer, intended for testing.
Because the model has a minimal config, we load the config to determine
vocabulary size and generate synthetic inputs directly.
"""
import torch
from transformers import AutoModelForMaskedLM, AutoConfig
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
    """Available Dummy Tokenizer Fast model variants."""

    DUMMY_TOKENIZER_FAST = "Dummy_Tokenizer_Fast"


class ModelLoader(ForgeModel):
    """Dummy Tokenizer Fast (ALBERT) For Masked LM model loader implementation for masked language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.DUMMY_TOKENIZER_FAST: ModelConfig(
            pretrained_model_name="robot-test/dummy-tokenizer-fast-with-model-config",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DUMMY_TOKENIZER_FAST

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.config = None

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
            model="Dummy_Tokenizer_Fast",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_config(self):
        """Load model config for the current variant.

        Returns:
            The loaded config instance
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Dummy Tokenizer Fast model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype.

        Returns:
            torch.nn.Module: The model instance for masked language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.config is None:
            self._load_config()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Dummy Tokenizer Fast model.

        This model has a minimal config, so we generate synthetic input_ids
        within the valid vocabulary range rather than using a tokenizer.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) that can be fed to the model.
        """
        if self.config is None:
            self._load_config()

        seq_length = 8
        vocab_size = self.config.vocab_size

        # Generate random token IDs within the valid vocabulary range
        input_ids = torch.randint(0, vocab_size, (1, seq_length))
        attention_mask = torch.ones(1, seq_length, dtype=torch.long)

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def decode_output(self, outputs):
        """Helper method to decode model outputs for masked language modeling.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: String representation of the predicted token IDs
        """
        if isinstance(outputs, list):
            logits = outputs[0].logits if hasattr(outputs[0], "logits") else outputs[0]
        else:
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

        predicted_token_ids = logits[0].argmax(axis=-1)

        return f"Output token IDs: {predicted_token_ids.tolist()}"
