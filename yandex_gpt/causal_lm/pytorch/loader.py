# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
YandexGPT 5 Lite model loader implementation for causal language modeling.
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


class ModelVariant(StrEnum):
    """Available YandexGPT 5 Lite model variants for causal language modeling."""

    YANDEX_GPT_5_LITE_8B_INSTRUCT = "YandexGPT_5_Lite_8B_Instruct"
    YANDEX_GPT_5_LITE_8B_PRETRAIN = "YandexGPT_5_Lite_8B_Pretrain"


class ModelLoader(ForgeModel):
    """YandexGPT 5 Lite model loader implementation for causal language modeling tasks."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.YANDEX_GPT_5_LITE_8B_INSTRUCT: LLMModelConfig(
            pretrained_model_name="yandex/YandexGPT-5-Lite-8B-instruct",
            max_length=128,
        ),
        ModelVariant.YANDEX_GPT_5_LITE_8B_PRETRAIN: LLMModelConfig(
            pretrained_model_name="yandex/YandexGPT-5-Lite-8B-pretrain",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.YANDEX_GPT_5_LITE_8B_INSTRUCT

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
        group = ModelGroup.VULCAN

        return ModelInfo(
            model="YandexGPT 5 Lite",
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
        """Load and return the YandexGPT 5 Lite model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The YandexGPT 5 Lite model instance for causal language modeling.
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

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.config = model.config

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the YandexGPT 5 Lite model with this instance's variant settings.

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

        if self.variant == ModelVariant.YANDEX_GPT_5_LITE_8B_PRETRAIN:
            prompts = [self.sample_text]
        else:
            messages = [{"role": "user", "content": self.sample_text}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts = [text]

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        # Add batch dimension
        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def get_mesh_config(self, num_devices: int):
        """Return mesh shape and axis names for tensor parallel."""
        if self.config.num_attention_heads % num_devices == 0:
            mesh_shape = (1, num_devices)
        elif (
            self.config.num_attention_heads % (num_devices // 2) == 0
            and num_devices % 2 == 0
        ):
            mesh_shape = (2, num_devices // 2)
        else:
            raise ValueError(
                f"Cannot evenly distribute {self.config.num_attention_heads} heads "
                f"across {num_devices} devices"
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

    def load_config(self):
        """Load and return the configuration for the YandexGPT 5 Lite model variant.

        Returns:
            The configuration object for the YandexGPT 5 Lite model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
