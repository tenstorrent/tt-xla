# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mistral model loader implementation for causal language modeling
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Mistral model variants."""

    MISTRAL_7B = "7b"
    MISTRAL_7B_INSTRUCT_V03 = "7b_instruct_v03"
    MINISTRAL_3B = "ministral_3b_instruct"
    MINISTRAL_8B = "ministral_8b_instruct"


class ModelLoader(ForgeModel):
    """Mistral model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.MISTRAL_7B: ModelConfig(
            pretrained_model_name="mistralai/Mistral-7B-v0.1",
        ),
        ModelVariant.MISTRAL_7B_INSTRUCT_V03: ModelConfig(
            pretrained_model_name="mistralai/Mistral-7B-Instruct-v0.3",
        ),
        ModelVariant.MINISTRAL_3B: ModelConfig(
            pretrained_model_name="ministral/Ministral-3b-instruct",
        ),
        ModelVariant.MINISTRAL_8B: ModelConfig(
            pretrained_model_name="mistralai/Ministral-8B-Instruct-2410",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MISTRAL_7B

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

        if variant in [ModelVariant.MISTRAL_7B_INSTRUCT_V03, ModelVariant.MINISTRAL_8B]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="mistral",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {
            "padding_side": "left",
        }
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the Mistral model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The Mistral model instance for causal language modeling.
        """
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        # Load pre-trained model from HuggingFace
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Mistral model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors (input_ids, attention_mask) that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Set up sample input
        test_input = "How often does the letter r occur in Mistral?"

        # Tokenize input
        inputs = self.tokenizer.encode_plus(test_input, return_tensors="pt")

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    # TODO - Verify this function correct (was AI_GENERATED)
    def decode_output(self, outputs, dtype_override):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass

        Returns:
            str: Decoded next token text
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Get logits for the last token
        next_token_logits = outputs.logits[:, -1]
        next_token = next_token_logits.softmax(dim=-1).argmax()
        return self.tokenizer.decode([next_token])

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        if self._variant not in [
            ModelVariant.MINISTRAL_3B,
        ]:
            assert (
                self.config.num_attention_heads % mesh_shape[1] == 0
            ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        if self._variant in [
            ModelVariant.MINISTRAL_3B,
        ]:
            return None

        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        return shard_specs
