# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
HTML-Pruner-Phi causal language modeling loader
"""
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer

from ....base import ForgeModel
from ....config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)
from ....tools.utils import cast_input_to_type


class ModelVariant(StrEnum):
    HTML_PRUNER_PHI_3_8B = "HTML_Pruner_Phi_3.8B"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.HTML_PRUNER_PHI_3_8B: ModelConfig(
            pretrained_model_name="zstanjj/HTML-Pruner-Phi-3.8B",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HTML_PRUNER_PHI_3_8B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="HTML-Pruner-Phi",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, *, dtype_override=None, **kwargs):
        self._ensure_tokenizer()

        model_kwargs = {"use_cache": False, "trust_remote_code": True}
        model_kwargs |= kwargs

        model = AutoModelForCausalLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, prompt: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = [
            {
                "role": "user",
                "content": prompt
                or "Can you provide ways to eat combinations of bananas and dragonfruits?",
            }
        ]
        text = self.tokenizer.apply_chat_template(
            input_prompt, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
            attn_mask = cast_input_to_type(attn_mask, dtype_override)
        return [input_ids, attn_mask]
