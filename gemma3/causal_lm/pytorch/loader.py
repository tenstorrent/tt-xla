# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

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
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Gemma3 model variants for causal LM."""

    GEMMA_3_270M = "270M"
    GEMMA_3_270M_IT = "270M_Instruct"
    GEMMA_3_1B_PT = "1B_Pretrained"
    GEMMA_3_1B_IT = "1B_Instruct"
    GEMMA_3_1B_IT_UNSLOTH = "1B_Instruct_Unsloth"
    GEMMA_3_27B_IT = "27B_Instruct"
    GEMMA_3_4B_IT_BNB_4BIT = "4B_Instruct_bnb_4bit"
    GEMMA_3_1B_IT_AWQ_INT4 = "1B_Instruct_awq_int4"


class ModelLoader(ForgeModel):
    """Gemma3 model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_270M: LLMModelConfig(
            pretrained_model_name="google/gemma-3-270m",
            max_length=256,
        ),
        ModelVariant.GEMMA_3_270M_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-3-270m-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_3_1B_PT: LLMModelConfig(
            pretrained_model_name="google/gemma-3-1b-pt",
            max_length=256,
        ),
        ModelVariant.GEMMA_3_1B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-3-1b-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_3_1B_IT_UNSLOTH: LLMModelConfig(
            pretrained_model_name="unsloth/gemma-3-1b-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_3_27B_IT: LLMModelConfig(
            pretrained_model_name="google/gemma-3-27b-it",
            max_length=256,
        ),
        ModelVariant.GEMMA_3_4B_IT_BNB_4BIT: LLMModelConfig(
            pretrained_model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            max_length=256,
        ),
        ModelVariant.GEMMA_3_1B_IT_AWQ_INT4: LLMModelConfig(
            pretrained_model_name="gaunernst/gemma-3-1b-it-int4-awq",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_270M_IT

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        if variant in (
            ModelVariant.GEMMA_3_27B_IT,
            ModelVariant.GEMMA_3_4B_IT_BNB_4BIT,
            ModelVariant.GEMMA_3_1B_IT_AWQ_INT4,
        ):
            group = ModelGroup.VULCAN
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="Gemma 3",
            variant=variant,
            group=group,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current causal_lm variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Gemma3 causal_lm model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Gemma3 model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        model_kwargs = {}
        if self._variant == ModelVariant.GEMMA_3_1B_IT_AWQ_INT4:
            model_kwargs["device_map"] = "cpu"
        elif self._variant == ModelVariant.GEMMA_3_4B_IT_BNB_4BIT:
            model_kwargs["device_map"] = "cpu"
        else:
            model_kwargs["use_cache"] = False
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
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        max_new_tokens: int = 256,
        prompt: Optional[str] = None,
    ):
        """Load and return sample inputs for the Gemma3 model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        if self._variant in (
            ModelVariant.GEMMA_3_270M,
            ModelVariant.GEMMA_3_1B_PT,
            ModelVariant.GEMMA_3_34M,
        ):
            inputs = self.tokenizer(
                prompt or self.sample_text,
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
        else:
            input_prompt = [
                {
                    "role": "user",
                    "content": prompt or self.sample_text,
                }
            ]
            input_text = self.tokenizer.apply_chat_template(
                input_prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self.tokenizer(
                [input_text],
                return_tensors="pt",
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return [input_ids, attn_mask]

    def get_mesh_config(self, num_devices: int):
        """Get the mesh configuration for tensor parallel execution."""
        mesh_shape = (1, num_devices)
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        """Load the sharding specification for tensor parallel execution."""
        if self._variant not in (
            ModelVariant.GEMMA_3_12B_IT_ABLITERATED,
            ModelVariant.GEMMA_3_27B_IT,
        ):
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
