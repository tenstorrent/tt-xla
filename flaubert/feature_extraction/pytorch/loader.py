# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FlauBERT model loader implementation for feature extraction.
"""

from typing import Optional

from third_party.tt_forge_models.config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available FlauBERT model variants."""

    TINY_RANDOM = "optimum-intel-internal-testing/tiny-random-flaubert"


class ModelLoader(ForgeModel):
    """FlauBERT model loader for feature extraction."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-flaubert",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="FlauBERT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        if self.tokenizer is None:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self._variant_config.pretrained_model_name
            )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from transformers import FlaubertModel

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaubertModel.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = getattr(self._variant_config, "max_length", 128)

        inputs = self.tokenizer(
            "La capitale de la France est Paris.",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs
