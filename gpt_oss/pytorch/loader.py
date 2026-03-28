# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
gpt-oss model loader implementation for causal language modeling tasks.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import Mxfp4Config
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
    """Available gpt-oss model variants."""

    GPT_OSS_20B = "20B"
    GPT_OSS_120B = "120B"
    GPT_OSS_120B_BNB_4BIT = "120B_bnb_4bit"


class ModelLoader(ForgeModel):
    """gpt-oss model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.GPT_OSS_20B: LLMModelConfig(
            pretrained_model_name="openai/gpt-oss-20b",
            max_length=256,
        ),
        ModelVariant.GPT_OSS_120B: LLMModelConfig(
            pretrained_model_name="openai/gpt-oss-120b",
            max_length=256,
        ),
        ModelVariant.GPT_OSS_120B_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/gpt-oss-120b-unsloth-bnb-4bit",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.GPT_OSS_20B

    # Sample messages for inference
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]

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
        self.config = None
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant == ModelVariant.GPT_OSS_120B_BNB_4BIT:
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.RED

        return ModelInfo(
            model="GPT-OSS",
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
        self.tokenizer.pad_token = self.tokenizer.eos_token

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the gpt-oss model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16 as default.

        Returns:
            torch.nn.Module: The gpt-oss model instance for causal language modeling.
        """
        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load config with modifications
        self.load_config()

        # Prepare model kwargs
        model_kwargs = {
            "config": self.config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        # BnB variants have quantization config baked in; others use Mxfp4
        if self._variant == ModelVariant.GPT_OSS_120B_BNB_4BIT:
            model_kwargs["device_map"] = "cpu"
        else:
            quantization_config = Mxfp4Config(dequantize=True)
            model_kwargs["quantization_config"] = quantization_config

        # Set dtype - default to bfloat16 if not specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs |= kwargs

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the gpt-oss model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           This is currently not used as tokenized inputs are integers.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
        )
        if (
            hasattr(self.model.config, "sliding_window")
            and self.model.config.sliding_window is not None
        ):
            # if the model uses sliding window attention, match sliding window value to input size so it
            # does not go out of bounds when updating the cache
            # Issue: https://github.com/tenstorrent/tt-xla/issues/3186
            self.model.config.sliding_window = inputs["input_ids"].shape[1]

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get mesh configuration for tensor parallelism.

        Args:
            num_devices: Number of devices to use for tensor parallelism

        Returns:
            Tuple of (mesh_shape, axis_names)
        """
        # Support different mesh configurations based on number of devices
        if num_devices == 32:  # Galaxy
            mesh_shape = (4, 8)
        elif num_devices == 8:  # llmbox
            mesh_shape = (2, 4)
        else:
            raise ValueError(f"Gpt-oss is only supported on llmbox and galaxy")

        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load shard specifications for tensor parallelism.

        Args:
            model: The gpt-oss model instance

        Returns:
            Dictionary mapping model parameters to their shard specifications,
            or None if sharding is not needed for this variant
        """
        shard_specs = {}

        # Embedding and output layers
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.model.norm.weight] = ("batch",)

        # lm_head sharding causes 20B hang: https://github.com/tenstorrent/tt-xla/issues/3484
        if self._variant and self._variant in (
            ModelVariant.GPT_OSS_120B,
            ModelVariant.GPT_OSS_120B_BNB_4BIT,
        ):
            shard_specs[model.lm_head.weight] = ("model", "batch")
        else:
            shard_specs[model.lm_head.weight] = (None, None)

        # Apply tensor parallel sharding to each transformer layer
        for layer in model.model.layers:
            # Attention layer sharding
            # q_proj weight shape: [2880, 4096]
            # Sharded column-wise (head-parallel): [2880, 4096/num_devices]
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)

            # k_proj weight shape: [2880, 512]
            # Sharded column-wise (head-parallel): [2880, 512/num_devices]
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)

            # v_proj weight shape: [2880, 512]
            # Sharded column-wise (head-parallel): [2880, 512/num_devices]
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)

            # o_proj weight shape: [4096, 2880]
            # Sharded row-wise: [4096/num_devices, 2880]
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.o_proj.bias] = ("batch",)

            # sinks shape: [4096]
            # Local replication per device (row-wise)
            shard_specs[layer.self_attn.sinks] = (None,)

            # MLP layer sharding
            # Router weight is replicated with batch sharding
            shard_specs[layer.mlp.router.weight] = (None, "batch")

            # Shard experts across devices
            # For example: 32 experts / 8 devices = 4 experts per device
            # [num_experts, hidden_size, 2 * expert_dim]
            shard_specs[layer.mlp.experts.gate_up_proj] = ("model", "batch", None)
            # [num_experts, 2 * expert_dim]
            shard_specs[layer.mlp.experts.gate_up_proj_bias] = ("model", None)
            # [num_experts, expert_dim, hidden_size]
            shard_specs[layer.mlp.experts.down_proj] = ("model", None, "batch")
            # [num_experts, hidden_size]
            shard_specs[layer.mlp.experts.down_proj_bias] = ("model", "batch")

            # Layer normalization weights
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the gpt-oss model with this instance's variant.

        Returns:
            The configuration object for the gpt-oss model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, trust_remote_code=True
        )
        if self.num_layers is not None:
            self.config.num_hidden_layers = self.num_layers

        return self.config
