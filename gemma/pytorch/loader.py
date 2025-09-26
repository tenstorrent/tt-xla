# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma model loader implementation for causal language modeling.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional

from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.model_utils import pad_inputs
from ...tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    """Available Gemma model variants for causal LM."""

    # Gemma 1.x
    GEMMA_1_1_2B_IT = "google/gemma-1.1-2b-it"
    GEMMA_1_1_7B_IT = "google/gemma-1.1-7b-it"
    GEMMA_2B = "google/gemma-2b"

    # Gemma 2.x
    GEMMA_2_2B_IT = "google/gemma-2-2b-it"
    GEMMA_2_9B_IT = "google/gemma-2-9b-it"
    GEMMA_2_27B_IT = "google/gemma-2-27b-it"


class ModelLoader(ForgeModel):
    """Gemma model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_1_1_2B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_1_1_2B_IT),
            max_length=256,
        ),
        ModelVariant.GEMMA_1_1_7B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_1_1_7B_IT),
            max_length=256,
        ),
        ModelVariant.GEMMA_2B: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_2B),
            max_length=256,
        ),
        ModelVariant.GEMMA_2_2B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_2_2B_IT),
            max_length=256,
        ),
        ModelVariant.GEMMA_2_9B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_2_9B_IT),
            max_length=256,
        ),
        ModelVariant.GEMMA_2_27B_IT: LLMModelConfig(
            pretrained_model_name=str(ModelVariant.GEMMA_2_27B_IT),
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_1_1_2B_IT

    sample_text = "What is your favorite city?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.seq_len = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        # Instruct and larger models are RED, others generality
        if any(x in variant.value for x in ["it", "7b", "9b", "27b"]):
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="gemma_causal_lm",
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

    def load_model(self, dtype_override=None):
        """Load and return the Gemma model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Gemma model instance for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
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
        """Load and return sample inputs for the Gemma model with default settings.

        Returns:
            dict: Input tensors and attention masks that can be fed to the model.
        """
        max_length = self._variant_config.max_length
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)
        input_prompt = prompt or self.sample_text
        inputs = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        if dtype_override is not None:
            for key in inputs:
                inputs[key] = cast_input_to_type(inputs[key], dtype_override)
        padded_input_ids, seq_len = pad_inputs(inputs["input_ids"], max_new_tokens)
        padded_attention_mask, _ = pad_inputs(inputs["attention_mask"], max_new_tokens)
        self.seq_len = seq_len
        inputs["input_ids"] = padded_input_ids
        inputs["attention_mask"] = padded_attention_mask
        return inputs

    def get_mesh_config(self, num_devices: int):
        mesh_shape = (1, num_devices)
        if self._variant not in [
            ModelVariant.GEMMA_1_1_2B_IT,
            ModelVariant.GEMMA_2B,
            ModelVariant.GEMMA_2_2B_IT,
        ]:
            assert (
                self.config.num_attention_heads % mesh_shape[1] == 0
            ), "Attention heads must be divisible by the model axis size"
        return mesh_shape, ("batch", "model")

    def load_shard_spec(self, model):
        if self._variant in [
            ModelVariant.GEMMA_1_1_2B_IT,
            ModelVariant.GEMMA_2B,
            ModelVariant.GEMMA_2_2B_IT,
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
