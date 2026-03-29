# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 LogiQA DPO LoRA model loader implementation for causal language modeling.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
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
    """Available Gemma3 LogiQA DPO LoRA model variants."""

    GEMMA_3_4B_IT_LOGIQA_DPO = "4B_Instruct_LogiQA_DPO"


class ModelLoader(ForgeModel):
    """Gemma3 LogiQA DPO LoRA model loader for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.GEMMA_3_4B_IT_LOGIQA_DPO: ModelConfig(
            pretrained_model_name="qiaw99/Gemma3-4b-it-LogiQA-DPO-F",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.GEMMA_3_4B_IT_LOGIQA_DPO

    BASE_MODEL_NAME = "google/gemma-3-4b-it"

    sample_text = "What is your favorite city?"

    def __init__(
        self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None
    ):
        super().__init__(variant)
        self.tokenizer = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Gemma 3 LogiQA DPO LoRA",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.BASE_MODEL_NAME, **tokenizer_kwargs
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(self.BASE_MODEL_NAME)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model_kwargs |= kwargs

        base_model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_NAME, **model_kwargs
        )

        adapter_name = self._variant_config.pretrained_model_name
        model = PeftModel.from_pretrained(base_model, adapter_name)
        model = model.merge_and_unload()

        for param in model.parameters():
            param.requires_grad = False

        return model

    def load_inputs(
        self,
        dtype_override=None,
        batch_size=1,
        prompt: Optional[str] = None,
    ):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

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
            max_length=256,
            padding="max_length",
            truncation=True,
        )
        for key in inputs:
            inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return [inputs["input_ids"], inputs["attention_mask"]]
