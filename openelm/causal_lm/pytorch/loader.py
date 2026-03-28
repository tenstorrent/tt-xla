# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenELM causal language modeling loader
"""
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

from ....config import (
    ModelConfig,
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
    OPENELM_270M_INSTRUCT = "270M_Instruct"
    OPENELM_450M_INSTRUCT = "450M_Instruct"
    OPENELM_1_1B_INSTRUCT = "1_1B_Instruct"
    OPENELM_3B_INSTRUCT = "3B_Instruct"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.OPENELM_270M_INSTRUCT: ModelConfig(
            pretrained_model_name="apple/OpenELM-270M-Instruct",
        ),
        ModelVariant.OPENELM_450M_INSTRUCT: ModelConfig(
            pretrained_model_name="apple/OpenELM-450M-Instruct",
        ),
        ModelVariant.OPENELM_1_1B_INSTRUCT: ModelConfig(
            pretrained_model_name="apple/OpenELM-1.1B-Instruct",
        ),
        ModelVariant.OPENELM_3B_INSTRUCT: ModelConfig(
            pretrained_model_name="apple/OpenELM-3B-Instruct",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.OPENELM_3B_INSTRUCT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="OpenELM",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            # OpenELM uses the LLaMA 2 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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
        text = prompt or "Once upon a time there was"
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
