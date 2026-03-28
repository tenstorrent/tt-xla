# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
BART model loader implementation for conditional generation (Chinese).
"""

import torch
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
    """Available BART model variants."""

    BASE_CHINESE = "Base_Chinese"


class ModelLoader(ForgeModel):
    """BART model loader implementation for conditional generation."""

    _VARIANTS = {
        ModelVariant.BASE_CHINESE: LLMModelConfig(
            pretrained_model_name="OpenMOSS-Team/bart-base-chinese",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_CHINESE

    sample_text = "北京是中国的首都。"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BART",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        from transformers import BertTokenizer

        self._tokenizer = BertTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import BartForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BartForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Seq2seq models need decoder_input_ids for the forward pass.
        decoder_start_token_id = (
            self._tokenizer.cls_token_id or self._tokenizer.bos_token_id
        )
        inputs["decoder_input_ids"] = torch.tensor([[decoder_start_token_id]])

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
