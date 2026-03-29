# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MetricX-24 model loader implementation for translation quality estimation.
"""

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration
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
    """Available MetricX-24 model variants."""

    HYBRID_LARGE_V2P6 = "hybrid-large-v2p6"


class ModelLoader(ForgeModel):
    """MetricX-24 model loader implementation for translation quality estimation."""

    _VARIANTS = {
        ModelVariant.HYBRID_LARGE_V2P6: ModelConfig(
            pretrained_model_name="google/metricx-24-hybrid-large-v2p6-bfloat16",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HYBRID_LARGE_V2P6

    # Sample source-hypothesis pair for reference-free quality estimation
    sample_source = "The quick brown fox jumps over the lazy dog."
    sample_hypothesis = "Le rapide renard brun saute par-dessus le chien paresseux."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="MetricX-24",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MT5ForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # MetricX uses "candidate: {hypothesis} source: {source}" format for QE
        input_text = f"candidate: {self.sample_hypothesis} source: {self.sample_source}"

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        # MetricX uses a single dummy decoder token (token ID 0) for regression
        inputs["decoder_input_ids"] = torch.zeros((1, 1), dtype=torch.long)

        return inputs
