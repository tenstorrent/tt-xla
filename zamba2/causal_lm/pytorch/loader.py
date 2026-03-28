# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional
import torch

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
from ....tools.utils import (
    pad_inputs,
    cast_input_to_type,
    get_static_cache_decode_inputs,
)


class ModelVariant(StrEnum):
    """Available model variants for causal LM."""

    ZAMBA2_7B_INSTRUCT = "Zamba2-7B-Instruct"


class ModelLoader(ForgeModel):
    """Model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.ZAMBA2_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Zyphra/Zamba2-7B-Instruct",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ZAMBA2_7B_INSTRUCT

    # Sample text for causal LM
    sample_text = "Hey how are you doing today?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            num_layers: Optional number of hidden layers to use. If None, uses the model's default.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.config = None
        self.num_layers = num_layers

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
            model="Zamba2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )

        # Set pad token to eos token for models
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The model instance for causal LM.
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if self.num_layers is not None:
            model.model.layers = model.model.layers[: self.num_layers]

        model.eval()
        self.model = model
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors suitable for causal LM.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # For causal LM, we need both input_ids and attention_mask
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Replicate tensors for batch size
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)

        # Pad input_ids and attention_mask
        target_len = self._variant_config.max_length
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], target_len)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], target_len)
        self.seq_len = seq_len

        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        """Load decode-step inputs (single token + static KV cache).
        Attention mask is intentionally omitted for single-batch decode. Defaults to steady-state decode.
        """

        # Ensure tokenizer and config are initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        max_cache_len = self._variant_config.max_length
        self.seq_len = 1

        return get_static_cache_decode_inputs(
            tokenizer=self.tokenizer,
            config=self.config,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype_override,
        )

    def decode_output(self, max_new_tokens, model, inputs, tokenizer):
        """Generates text .
        Args:
            max_new_tokens (int): The maximum number of new tokens to generate.
            model (torch.nn.Module): The language model used for token generation.
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
            tokenizer: The tokenizer used to decode token IDs into text.
        """
        current_pos = self.seq_len

        for _ in range(max_new_tokens):
            logits = model(*inputs)

            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            next_token_logits = logits[:, current_pos - 1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1)

            if next_token_id.item() == tokenizer.eos_token_id:
                break

            # Update input_ids and attention_mask
            inputs[0][:, current_pos] = next_token_id
            inputs[1][:, current_pos] = 1

            current_pos += 1

        valid_tokens = inputs[0][:, self.seq_len : current_pos].view(-1).tolist()
        answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)
        return answer

    def load_config(self):
        """Load and return the configuration for the model variant.

        Returns:
            The configuration object for the model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
