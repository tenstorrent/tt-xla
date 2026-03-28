# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 2.5 Coder model loader implementation
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Qwen 2.5 Coder model variants for causal language modeling."""

    QWEN_2_5_CODER_0_5B = "0.5B"
    QWEN_2_5_CODER_1_5B = "1.5B"
    QWEN_2_5_CODER_1_5B_INSTRUCT = "1.5B_Instruct"
    QWEN_2_5_CODER_1_5B_INSTRUCT_UNSLOTH = "1.5B_Instruct_Unsloth"
    QWEN_2_5_CODER_3B = "3B"
    QWEN_2_5_CODER_3B_INSTRUCT = "3B_Instruct"
    QWEN_2_5_CODER_7B = "7B"
    QWEN_2_5_CODER_7B_INSTRUCT = "7B_Instruct"
    QWEN_2_5_CODER_0_5B_INSTRUCT_BNB_4BIT = "0.5B_Instruct_bnb_4bit"
    QWEN_2_5_CODER_7B_BNB_4BIT = "7B_bnb_4bit"
    QWEN_2_5_CODER_14B_BNB_4BIT = "14B_bnb_4bit"
    QWEN_2_5_CODER_7B_INSTRUCT_GPTQ_INT4 = "7B_Instruct_GPTQ_Int4"
    QWEN_2_5_CODER_7B_INSTRUCT_MLX_4BIT = "7B_Instruct_MLX_4bit"
    QWEN_2_5_CODER_32B_INSTRUCT = "32B_Instruct"
    QWEN_2_5_CODER_3B_INSTRUCT_4BIT = "3B_Instruct_4bit"
    QWEN_2_5_CODER_32B_INSTRUCT_AWQ = "32B_Instruct_Awq"
    QWEN_2_5_CODER_32B_INSTRUCT_MLX_8BIT = "32B_Instruct_MLX_8bit"


class ModelLoader(ForgeModel):
    """Qwen 2.5 Coder model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_2_5_CODER_0_5B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-0.5B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_1_5B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-1.5B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT_UNSLOTH: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen2.5-Coder-1.5B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-3B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_3B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_7B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-7B",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT_GPTQ_INT4: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT_MLX_4BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2.5-Coder-7B-Instruct-MLX-4bit",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT_AWQ: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            max_length=128,
        ),
        ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT_MLX_8BIT: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_2_5_CODER_0_5B

    # Shared configuration parameters
    sample_text = "write a quick sort algorithm."

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
        group = ModelGroup.GENERALITY
        if variant == ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT:
            group = ModelGroup.RED
        if variant in [
            ModelVariant.QWEN_2_5_CODER_0_5B_INSTRUCT_BNB_4BIT,
            ModelVariant.QWEN_2_5_CODER_1_5B_INSTRUCT_UNSLOTH,
            ModelVariant.QWEN_2_5_CODER_7B_BNB_4BIT,
            ModelVariant.QWEN_2_5_CODER_14B_BNB_4BIT,
            ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT_GPTQ_INT4,
            ModelVariant.QWEN_2_5_CODER_7B_INSTRUCT_MLX_4BIT,
            ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT_AWQ,
            ModelVariant.QWEN_2_5_CODER_32B_INSTRUCT_MLX_8BIT,
        ]:
            group = ModelGroup.VULCAN

        return ModelInfo(
            model="Qwen 2.5 Coder",
            variant=variant,
            group=group,
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
        """Load and return the Qwen 2.5 Coder model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 2.5 Coder model instance for causal language modeling.
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

        # Quantized variants need device_map="cpu" for CPU-based loading
        if pretrained_model_name in (
            "unsloth/Qwen2.5-Coder-0.5B-Instruct-bnb-4bit",
            "unsloth/Qwen2.5-Coder-7B-bnb-4bit",
            "unsloth/Qwen2.5-Coder-14B-bnb-4bit",
            "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4",
            "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
            "Qwen/Qwen2.5-Coder-32B-Instruct-AWQ",
            "lmstudio-community/Qwen2.5-Coder-32B-Instruct-MLX-8bit",
        ):
            model_kwargs["device_map"] = "cpu"
        if "mlx-community" in pretrained_model_name:
            model_kwargs["ignore_mismatched_sizes"] = True

        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        # Store config for mesh/sharding validation
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 2.5 Coder model with this instance's variant settings.

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

        messages = [
            {
                "role": "system",
                "content": "You are Qwen, created by TT Cloud. You are a helpful assistant.",
            },
            {"role": "user", "content": self.sample_text},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts = [text]

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
            num_devices % 2 == 0
            and self.config.num_attention_heads % (num_devices // 2) == 0
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
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs
