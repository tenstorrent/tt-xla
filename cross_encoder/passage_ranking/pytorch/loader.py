# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Cross-Encoder model loader implementation for passage ranking.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Cross-Encoder model variants for passage ranking."""

    MS_MARCO_MINILM_L2_V2 = "ms-marco-MiniLM-L2-v2"
    MS_MARCO_MINILM_L4_V2 = "ms-marco-MiniLM-L4-v2"
    MS_MARCO_MINILM_L6_V2 = "ms-marco-MiniLM-L6-v2"
    MS_MARCO_MINILM_L12_V2 = "ms-marco-MiniLM-L12-v2"
    MS_MARCO_TINYBERT_L6 = "ms-marco-TinyBERT-L6"
    MSMARCO_MINILM_L12_EN_DE_V1 = "msmarco-MiniLM-L12-en-de-v1"


class ModelLoader(ForgeModel):
    """Cross-Encoder model loader implementation for passage ranking."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.MS_MARCO_MINILM_L2_V2: ModelConfig(
            pretrained_model_name="cross-encoder/ms-marco-MiniLM-L2-v2",
        ),
        ModelVariant.MS_MARCO_MINILM_L4_V2: ModelConfig(
            pretrained_model_name="cross-encoder/ms-marco-MiniLM-L4-v2",
        ),
        ModelVariant.MS_MARCO_MINILM_L6_V2: ModelConfig(
            pretrained_model_name="cross-encoder/ms-marco-MiniLM-L6-v2",
        ),
        ModelVariant.MS_MARCO_MINILM_L12_V2: ModelConfig(
            pretrained_model_name="cross-encoder/ms-marco-MiniLM-L12-v2",
        ),
        ModelVariant.MS_MARCO_TINYBERT_L6: ModelConfig(
            pretrained_model_name="cross-encoder/ms-marco-TinyBERT-L6",
        ),
        ModelVariant.MSMARCO_MINILM_L12_EN_DE_V1: ModelConfig(
            pretrained_model_name="cross-encoder/msmarco-MiniLM-L12-en-de-v1",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MS_MARCO_MINILM_L6_V2

    # Sample query-passage pairs for testing
    sample_pairs = [
        (
            "How many people live in Berlin?",
            "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.",
        ),
    ]

    # Variant-specific sample pairs (e.g., for non-English models)
    _VARIANT_SAMPLE_PAIRS = {
        ModelVariant.MMARCO_GERMAN_DISTILBERT_BASE: [
            (
                "Wie viele Menschen leben in Berlin?",
                "Berlin hatte mass 3.520.031 registrierte Einwohner auf einer Flaeche von 891,82 Quadratkilometern.",
            ),
        ],
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="CrossEncoder",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cross-Encoder model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The Cross-Encoder model instance for passage ranking.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Cross-Encoder model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        pairs = self._VARIANT_SAMPLE_PAIRS.get(self._variant, self.sample_pairs)
        queries = [pair[0] for pair in pairs]
        passages = [pair[1] for pair in pairs]

        inputs = self.tokenizer(
            queries,
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
