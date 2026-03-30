# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Olmo3 Causal LM model loader implementation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Olmo3 model variants for causal language modeling."""

    Olmo_3_7B_Think = "3_7b_think"
    Olmo_3_7B_Think_SFT = "3_7b_think_sft"
    Olmo_3_7B_Instruct = "3_7b_instruct"
    Olmo_3_7B_Instruct_SFT = "3_7b_instruct_sft"
    Olmo_3_1025_7B = "3_1025_7b"
    Olmo_3_32B_Think = "3_32b_think"
    Olmo_3_1125_32B = "3_1125_32b"
    Unsloth_Olmo_3_7B_Instruct = "unsloth_3_7b_instruct"


class ModelLoader(ForgeModel):
    """Olmo3 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.Olmo_3_7B_Think: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-7B-Think",
            max_length=256,
        ),
        ModelVariant.Olmo_3_7B_Think_SFT: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-7B-Think-SFT",
            max_length=256,
        ),
        ModelVariant.Olmo_3_7B_Instruct: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-7B-Instruct",
            max_length=256,
        ),
        ModelVariant.Olmo_3_7B_Instruct_SFT: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-7B-Instruct-SFT",
            max_length=256,
        ),
        ModelVariant.Olmo_3_1025_7B: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-1025-7B",
            max_length=256,
        ),
        ModelVariant.Olmo_3_1125_32B: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-1125-32B",
            max_length=256,
        ),
        ModelVariant.Olmo_3_32B_Think: LLMModelConfig(
            pretrained_model_name="allenai/Olmo-3-32B-Think",
            max_length=256,
        ),
        ModelVariant.Unsloth_Olmo_3_7B_Instruct: LLMModelConfig(
            pretrained_model_name="unsloth/Olmo-3-7B-Instruct",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.Olmo_3_7B_Think

    # Shared configuration parameters
    sample_text = "Who would win in a fight - a dinosaur or a cow named Moo Moo?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        if variant == ModelVariant.Unsloth_Olmo_3_7B_Instruct:
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.RED
        return ModelInfo(
            model="olmo_3",
            variant=variant,
            group=variant_groups.get(variant, ModelGroup.RED),
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
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Olmo 3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Olmo 3 model instance for causal language modeling.
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

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        if getattr(model.config, "use_cache", True):
            model.config.layer_types = [
                "full_attention"
            ] * model.config.num_hidden_layers
        model.eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Olmo 3 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        prompts = [self.sample_text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def get_mesh_config(self, num_devices: int):

        # Prefer (1, N) when heads divide N, otherwise try (2, N/2)
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads across {num_devices} devices"
            )
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("batch", "model")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Olmo 3 model variant.

        Returns:
            The configuration object for the Olmo 3 model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
