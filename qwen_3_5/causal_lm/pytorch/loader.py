# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Qwen 3.5 model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
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
from ....tools.utils import get_static_cache_decode_inputs


class ModelVariant(StrEnum):
    """Available Qwen 3.5 model variants for causal language modeling."""

    QWEN_3_5_0_8B = "0.8B"
    QWEN_3_5_2B = "2B"
    QWEN_3_5_2B_BASE = "2B_BASE"
    QWEN_3_5_4B = "4B"
    QWEN_3_5_4B_BASE = "4B_Base"
    QWEN_3_5_9B = "9B"
    QWEN_3_5_9B_BASE = "9B_Base"
    QWEN_3_5_27B = "27B"
    QWEN_3_5_27B_FP8 = "27B_FP8"
    QWEN_3_5_27B_NVFP4 = "27B_NVFP4"
    QWEN_3_5_35B_A3B = "35B_A3B"
    QWEN_3_5_35B_A3B_BASE = "35B_A3B_Base"
    QWEN_3_5_35B_A3B_FP8 = "35B_A3B_FP8"
    QWEN_3_5_0_8B_GGUF = "0.8B_GGUF"
    QWEN_3_5_4B_GGUF = "4B_GGUF"
    QWEN_3_5_9B_GGUF = "9B_GGUF"
    QWEN_3_5_4B_LMSTUDIO_GGUF = "4B_LMStudio_GGUF"
    QWEN_3_5_27B_TEXT_NVFP4_MTP = "27B_Text_NVFP4_MTP"
    QWEN_3_5_35B_A3B_NVFP4 = "35B_A3B_NVFP4"
    QWEN_3_5_397B_A17B_GGUF = "397B_A17B_GGUF"
    QWEN_3_5_27B_CLAUDE_OPUS_DISTILLED_MLX_6BIT = "27B_Claude_Opus_Distilled_MLX_6bit"
    QWEN_3_5_122B_A10B_MXFP4 = "122B_A10B_MXFP4"


class ModelLoader(ForgeModel):
    """Qwen 3.5 model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.QWEN_3_5_0_8B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-0.8B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_2B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-2B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_2B_BASE: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-2B-Base",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_4B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-4B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_4B_BASE: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-4B-Base",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_9B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-9B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_9B_BASE: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-9B-Base",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-27B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-27B-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_NVFP4: LLMModelConfig(
            pretrained_model_name="apolo13x/Qwen3.5-27B-NVFP4",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-35B-A3B",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B_BASE: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-35B-A3B-Base",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B_FP8: LLMModelConfig(
            pretrained_model_name="Qwen/Qwen3.5-35B-A3B-FP8",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_0_8B_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen3.5-0.8B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_4B_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3.5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_4B_LMSTUDIO_GGUF: LLMModelConfig(
            pretrained_model_name="lmstudio-community/Qwen3.5-4B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_9B_GGUF: LLMModelConfig(
            pretrained_model_name="unsloth/Qwen3.5-9B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_TEXT_NVFP4_MTP: LLMModelConfig(
            pretrained_model_name="osoleve/Qwen3.5-27B-Text-NVFP4-MTP",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_35B_A3B_NVFP4: LLMModelConfig(
            pretrained_model_name="AxionML/Qwen3.5-35B-A3B-NVFP4",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_397B_A17B_GGUF: LLMModelConfig(
            pretrained_model_name="AesSedai/Qwen3.5-397B-A17B-GGUF",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_27B_CLAUDE_OPUS_DISTILLED_MLX_6BIT: LLMModelConfig(
            pretrained_model_name="mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-6bit",
            max_length=128,
        ),
        ModelVariant.QWEN_3_5_122B_A10B_MXFP4: LLMModelConfig(
            pretrained_model_name="olka-fi/Qwen3.5-122B-A10B-MXFP4",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.QWEN_3_5_9B

    # GGUF files for quantized variants
    _GGUF_FILES = {
        ModelVariant.QWEN_3_5_0_8B_GGUF: "Qwen3.5-0.8B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_5_4B_GGUF: "Qwen3.5-4B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_5_4B_LMSTUDIO_GGUF: "Qwen3.5-4B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_5_9B_GGUF: "Qwen3.5-9B-Q4_K_M.gguf",
        ModelVariant.QWEN_3_5_397B_A17B_GGUF: "Qwen3.5-397B-A17B-Q4_K_M-00001-of-00007.gguf",
    }

    # Shared configuration parameters
    sample_text = "Give me a short introduction to large language model."

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
        self.config = None
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
            model="Qwen 3.5",
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
        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        # Pass gguf_file for GGUF variants
        if self._is_gguf_variant():
            tokenizer_kwargs["gguf_file"] = self._gguf_file

        # Custom architecture variants require trust_remote_code
        if self._needs_remote_code():
            tokenizer_kwargs["trust_remote_code"] = True

        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Qwen 3.5 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Qwen 3.5 model instance for causal language modeling.
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

        # Pass gguf_file for GGUF variants
        if self._is_gguf_variant():
            model_kwargs["gguf_file"] = self._gguf_file

        # Custom architecture variants require trust_remote_code
        if self._needs_remote_code():
            model_kwargs["trust_remote_code"] = True

        # GPTQ variants need device_map="cpu" for CPU-based loading
        if self._variant in (ModelVariant.QWEN_3_5_397B_A17B_GPTQ_INT4,):
            model_kwargs["device_map"] = "cpu"

        # AutoRound int4 variants need device_map="cpu" for CPU-based loading
        if self._variant == ModelVariant.QWEN_3_5_35B_A3B_INT4_AUTOROUND:
            model_kwargs["device_map"] = "cpu"

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            if hasattr(config, "text_config"):
                config.text_config.num_hidden_layers = self.num_layers
                # Truncate layer_types to match the reduced number of layers
                if hasattr(config.text_config, "layer_types"):
                    config.text_config.layer_types = config.text_config.layer_types[
                        : self.num_layers
                    ]
            else:
                config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        ).eval()

        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Qwen 3.5 model with this instance's variant settings.

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

        # Base models use plain text; chat models use chat template
        if self._variant in (ModelVariant.QWEN_3_5_4B_BASE,):
            prompts = [self.sample_text]
        else:
            messages = [{"role": "user", "content": self.sample_text}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
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

    def _get_text_config(self):
        """Get the text config, handling both nested (MoE) and flat config structures."""
        if hasattr(self.config, "text_config"):
            return self.config.text_config
        return self.config

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        text_config = self._get_text_config()
        assert (
            text_config.num_attention_heads % mesh_shape[1] == 0
        ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def _is_gguf_variant(self):
        """Check if the current variant uses GGUF quantization."""
        return self._variant in self._GGUF_FILES

    @property
    def _gguf_file(self):
        """Get the GGUF filename for the current variant."""
        return self._GGUF_FILES.get(self._variant)

    def _is_awq_variant(self):
        """Check if the current variant uses AWQ quantization."""
        return False

    def _needs_remote_code(self):
        """Check if the current variant requires trust_remote_code for custom architecture."""
        return self._variant in (ModelVariant.QWEN_3_5_397B_A17B_GPTQ_INT4,)

    def _is_moe_variant(self):
        """Check if the current variant is a Mixture of Experts model."""
        return self._variant in (
            ModelVariant.QWEN_3_5_35B_A3B,
            ModelVariant.QWEN_3_5_35B_A3B_BASE,
            ModelVariant.QWEN_3_5_35B_A3B_BASE_UNSLOTH,
            ModelVariant.QWEN_3_5_35B_A3B_FP8,
            ModelVariant.QWEN_3_5_122B_A10B,
            ModelVariant.QWEN_3_5_122B_A10B_MXFP4,
            ModelVariant.QWEN_3_5_397B_A17B_GGUF,
        )

    def load_shard_spec(self, model):
        shard_specs = {}
        for layer in model.model.layers:
            if self._is_moe_variant():
                # MoE layers use fused expert weights (3D tensors)
                mlp = layer.mlp
                if hasattr(mlp, "experts"):
                    shard_specs[mlp.experts.gate_up_proj] = (None, "model", "batch")
                    shard_specs[mlp.experts.down_proj] = (None, "batch", "model")
                if hasattr(mlp, "shared_expert"):
                    shard_specs[mlp.shared_expert.up_proj.weight] = ("model", "batch")
                    shard_specs[mlp.shared_expert.gate_proj.weight] = (
                        "model",
                        "batch",
                    )
                    shard_specs[mlp.shared_expert.down_proj.weight] = (
                        "batch",
                        "model",
                    )
                # Layers have either self_attn (full attention) or linear_attn
                if hasattr(layer, "self_attn"):
                    shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                    shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                    shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                    shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
            else:
                shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
                shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

                shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
                shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[model.lm_head.weight] = ("model", "batch")

        return shard_specs

    def load_config(self):
        """Load and return the configuration for the Qwen 3.5 model variant.

        Returns:
            The configuration object for the Qwen 3.5 model.
        """
        config_kwargs = {}
        if self._is_gguf_variant():
            config_kwargs["gguf_file"] = self._gguf_file
        if self._needs_remote_code():
            config_kwargs["trust_remote_code"] = True

        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name, **config_kwargs
        )

        return self.config

    def load_inputs_decode(self, dtype_override=None, batch_size=1):
        """Load decode-step inputs (single token + static KV cache).
        Attention mask is intentionally omitted for single-batch decode. Defaults to steady-state decode.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if self.config is None:
            self.load_config()

        max_cache_len = getattr(self._variant_config, "max_length", None) or 128
        self.seq_len = 1

        return get_static_cache_decode_inputs(
            tokenizer=self.tokenizer,
            config=self.config,
            model=self.model,
            batch_size=batch_size,
            max_cache_len=max_cache_len,
            dtype=dtype_override,
        )
