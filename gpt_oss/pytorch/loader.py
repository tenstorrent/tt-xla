# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
gpt-oss model loader implementation for causal language modeling tasks.
"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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

    GPT_OSS_20B = "gpt_oss_20b"
    GPT_OSS_120B = "gpt_oss_120b"


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
        return ModelInfo(
            model="gpt_oss",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(
        self,
        dtype_override=None,
        mlp_type="a2a_sparse",
        num_devices=8,
        cluster_axis=0,
        flat_device_order=None,
    ):
        """Load and return the gpt-oss model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16 as default.
            mlp_type: MLP implementation to use. One of "original", "sparse", "a2a_sparse".
            num_devices: Number of devices for A2aSparseMLP dispatch/combine.
            cluster_axis: Mesh axis for A2aSparseMLP dispatch/combine.
            flat_device_order: Optional device order permutation for A2aSparseMLP.

        Returns:
            torch.nn.Module: The gpt-oss model instance for causal language modeling.
        """
        # Load config with modifications
        self.load_config()

        # Prepare model kwargs
        model_kwargs = {
            "config": self.config,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        # Set dtype - default to bfloat16 if not specified
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.mlp_type = mlp_type
        # Replace MLP layers if requested
        if mlp_type == "sparse":
            import tt_torch.sparse_mlp
            
            tt_torch.sparse_mlp.enable_sparse_mlp(model)
        elif mlp_type == "a2a_sparse":
            from tt_torch.sparse_mlp import A2aSparseMLP

            for layer in model.model.layers:
                layer.mlp = A2aSparseMLP(
                    layer.mlp,
                    num_experts=self.config.num_local_experts,
                    num_experts_per_tok=self.config.num_experts_per_tok,
                    num_devices=8,
                    dispatch_devices=2,
                    cluster_axis=0,
                    # reduce_axis=1,
                    config=self.config,
                    # flat_device_order=flat_device_order,
                )

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
            self._load_tokenizer()

        # Create tokenized inputs (single sample)
        inputs = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            max_length=128,
        )

        # Repeat to batch_size=4
        # inputs = {k: v.repeat(4, 1) for k, v in inputs.items()}

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Get mesh configuration for tensor parallelism.

        Args:
            num_devices: Number of devices to use for tensor parallelism

        Returns:
            Tuple of (mesh_shape, axis_names)
        """
        mesh_shape = (2, 4)
        return mesh_shape, ("model", "batch")

    def load_shard_spec(self, model):
        """Load shard specifications for tensor parallelism.

        Args:
            model: The gpt-oss model instance

        Returns:
            Dictionary mapping model parameters to their shard specifications,
            or None if sharding is not needed for this variant
        """
        # Sharding Strategy:
        # - Mesh: ("model"=2, "batch"=4) = 8 devices
        # - hidden_dim is consistently sharded on "batch" axis (2-way TP)
        # - Expert dim is sharded on "model" axis (4-way EP)
        # - K dimension sharding triggers all-reduce automatically

        shard_specs = {}

        # ===== Embedding & LM Head =====
        # (vocab_size, hidden_dim) -> hidden/"batch" for consistent sharding
        shard_specs[model.lm_head.weight] = (None, "batch")
        shard_specs[model.model.embed_tokens.weight] = (None, "batch")
        shard_specs[model.model.norm.weight] = ("batch",)

        for layer in model.model.layers:
            # ===== Layer Norms =====
            # (hidden_dim,) -> hidden/"batch" for consistent sharding
            shard_specs[layer.input_layernorm.weight] = ("batch",)
            shard_specs[layer.post_attention_layernorm.weight] = ("batch",)
            # ===== Self-Attention =====
            # Q/K/V: (heads*head_dim, hidden) -> 2D sharding
            #   - output dim (heads): "model" (4-way)
            #   - input dim (hidden): "batch" (2-way, K shard -> all-reduce)
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer.self_attn.q_proj.bias] = ("model",)
            shard_specs[layer.self_attn.k_proj.bias] = ("model",)
            shard_specs[layer.self_attn.v_proj.bias] = ("model",)

            # O: (hidden, heads*head_dim) -> 2D sharding (reversed)
            #   - output dim (hidden): "batch" (2-way)
            #   - input dim (heads): "model" (4-way)
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            shard_specs[layer.self_attn.o_proj.bias] = (None,)
            shard_specs[layer.self_attn.sinks] = (None,)

            # ===== MoE MLP =====
            # Router: (num_experts, hidden) -> hidden/"batch" for K shard
            shard_specs[layer.mlp.router.weight] = (None, "batch")
            # shard_specs[layer.mlp.router.bias] = ("batch",)
            
            # if self.mlp_type is None or self.mlp_type == "sparse":
            if True:
                shard_specs[layer.mlp.experts.gate_up_proj] = (("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.gate_up_proj_bias] = (("model", "batch"), None)

                shard_specs[layer.mlp.experts.down_proj] = (("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.down_proj_bias] = (("model", "batch"), None)
            else :
                shard_specs[layer.mlp.experts.gate_up_proj] = (None, ("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.gate_up_proj_bias] = (("model", "batch"), None)

                shard_specs[layer.mlp.experts.down_proj] = (None, ("model", "batch"), None, None)
                shard_specs[layer.mlp.experts.down_proj_bias] = (("model", "batch"), None)
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

        self.config.quantization_config["quant_method"] = "none"
        self.config.use_cache = False
        return self.config
