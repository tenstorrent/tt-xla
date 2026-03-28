# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PubMedBERT model loader implementation for natural language inference.
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
    """Available PubMedBERT model variants for NLI."""

    MNLI_MEDNLI = "mnli-mednli"


class ModelLoader(ForgeModel):
    """PubMedBERT model loader implementation for natural language inference."""

    _VARIANTS = {
        ModelVariant.MNLI_MEDNLI: ModelConfig(
            pretrained_model_name="pritamdeka/PubMedBERT-MNLI-MedNLI",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MNLI_MEDNLI

    # Sample premise-hypothesis pairs for NLI testing
    sample_pairs = [
        (
            "The patient showed signs of acute respiratory distress.",
            "The patient has difficulty breathing.",
        ),
    ]

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
            model="PubMedBERT",
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
        """Load and return the PubMedBERT model instance for NLI.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The PubMedBERT model instance for NLI.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self.model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the PubMedBERT NLI model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        premises = [pair[0] for pair in self.sample_pairs]
        hypotheses = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            premises,
            hypotheses,
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

    def decode_output(self, outputs, inputs=None):
        """Helper method to decode model outputs into human-readable text.

        Args:
            outputs: Model output from a forward pass (logits)
            inputs: Optional input tensors used to generate the outputs

        Returns:
            str: Predicted NLI label
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        logits = outputs[0]
        predicted_class_id = logits.argmax().item()
        predicted_category = self.model.config.id2label[predicted_class_id]

        return predicted_category
