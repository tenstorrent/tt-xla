# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SpamNS v1 model loader implementation for sequence classification (Russian spam detection).
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from third_party.tt_forge_models.config import (
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
    """Available SpamNS v1 model variants for sequence classification."""

    SPAMNS_V1 = "SpamNS_v1"


class ModelLoader(ForgeModel):
    """SpamNS v1 model loader implementation for Russian spam detection."""

    _VARIANTS = {
        ModelVariant.SPAMNS_V1: LLMModelConfig(
            pretrained_model_name="RUSpam/spamNS_v1",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPAMNS_V1

    _SAMPLE_TEXTS = {
        ModelVariant.SPAMNS_V1: "Поздравляем! Вы выиграли приз! Перейдите по ссылке для получения.",
    }

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.review = self._SAMPLE_TEXTS.get(
            self._variant, "Поздравляем! Вы выиграли приз!"
        )
        self.max_length = 128
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="SpamNS_v1",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load SpamNS v1 model for sequence classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The SpamNS v1 model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for SpamNS v1 sequence classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for spam classification.

        Args:
            co_out: Model output
        """
        logits = co_out[0]
        probability = torch.sigmoid(logits).item()
        label = "spam" if probability > 0.5 else "not spam"
        print(f"Predicted: {label} (probability: {probability:.4f})")
