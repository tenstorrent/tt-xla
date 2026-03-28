# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MetricX-24 model loader implementation for translation quality evaluation.

MetricX-24 is a learned regression metric for machine translation quality
based on mT5. It outputs a score in [0, 25] (lower is better, MQM convention).

The MT5ForRegression class is provided by the metricx24 package
(https://github.com/google-research/metricx).

Available variants:
- HYBRID_XXL_V2P6: google/metricx-24-hybrid-xxl-v2p6-bfloat16
"""

from transformers import AutoTokenizer
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MetricX-24 model variants."""

    HYBRID_XXL_V2P6 = "Hybrid_XXL_v2p6"


class ModelLoader(ForgeModel):
    """MetricX-24 model loader for translation quality regression."""

    _VARIANTS = {
        ModelVariant.HYBRID_XXL_V2P6: LLMModelConfig(
            pretrained_model_name="google/metricx-24-hybrid-xxl-v2p6-bfloat16",
            max_length=512,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.HYBRID_XXL_V2P6

    # Sample input: reference-based translation quality evaluation
    sample_source = "The quick brown fox jumps over the lazy dog."
    sample_translation = "Le rapide renard brun saute par-dessus le chien paresseux."
    sample_reference = "Le renard brun rapide saute par-dessus le chien paresseux."

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="MetricX",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MetricX-24 MT5ForRegression model.

        Requires the metricx24 package:
            pip install git+https://github.com/google-research/metricx.git
        """
        from metricx24.models import MT5ForRegression

        pretrained_model_name = self._variant_config.pretrained_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MT5ForRegression.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for MetricX-24.

        Input is formatted as: "candidate: {translation} | reference: {reference}"
        for reference-based evaluation.
        """
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        input_text = (
            f"candidate: {self.sample_translation} | "
            f"reference: {self.sample_reference}"
        )

        inputs = self.tokenizer(
            input_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
