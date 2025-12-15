# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-3 token classification loader
"""
from typing import Optional

from transformers import AutoTokenizer, Phi3ForTokenClassification

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
    MINI_128K = "microsoft/Phi-3-mini-128k-instruct"
    MINI_4K = "microsoft/Phi-3-mini-4k-instruct"


class ModelLoader(ForgeModel):
    _VARIANTS = {
        ModelVariant.MINI_128K: ModelConfig(
            pretrained_model_name=str(ModelVariant.MINI_128K)
        ),
        ModelVariant.MINI_4K: ModelConfig(
            pretrained_model_name=str(ModelVariant.MINI_4K)
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MINI_128K

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="phi3_token_cls",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )

    def load_model(self, dtype_override=None):
        self._ensure_tokenizer()
        model = Phi3ForTokenClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            use_cache=False,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, text: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = [
            {
                "role": "user",
                "content": text
                or "Can you provide ways to eat combinations of bananas and dragonfruits?",
            }
        ]
        text = self.tokenizer.apply_chat_template(
            input_prompt, add_generation_prompt=True, tokenize=False
        )
        inputs = self.tokenizer([text], return_tensors="pt")
        input_ids = inputs["input_ids"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
        return [input_ids]
