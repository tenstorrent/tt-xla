# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Phi-3 sequence classification loader
"""
from typing import Optional

from transformers import AutoTokenizer, Phi3Config, Phi3ForSequenceClassification

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
            model="phi3_seq_cls",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name, trust_remote_code=True
            )
            # Set pad token if not already set (PHI models often need this)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, dtype_override=None):
        self._ensure_tokenizer()
        cfg = Phi3Config.from_pretrained(self._variant_config.pretrained_model_name)
        cfg_dict = cfg.to_dict()
        cfg_dict["use_cache"] = False
        cfg_dict["pad_token_id"] = self.tokenizer.pad_token_id  # Set to match tokenizer
        cfg = Phi3Config(**cfg_dict)

        model = Phi3ForSequenceClassification.from_pretrained(
            self._variant_config.pretrained_model_name,
            trust_remote_code=True,
            config=cfg,
        )
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None, text: Optional[str] = None):
        self._ensure_tokenizer()
        input_prompt = text or "the movie was great!"
        inputs = self.tokenizer(input_prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        if dtype_override is not None:
            input_ids = cast_input_to_type(input_ids, dtype_override)
        return [input_ids]
