# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT model loader implementation for token classification.
"""

import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizer
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
    """Available DistilBERT model variants for token classification."""

    DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL = (
        "Davlan/distilbert-base-multilingual-cased-ner-hrl"
    )


class ModelLoader(ForgeModel):
    """DistilBERT model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL: LLMModelConfig(
            pretrained_model_name="Davlan/distilbert-base-multilingual-cased-ner-hrl",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.max_length = 128
        self.tokenizer = None
        self.sample_text = "HuggingFace is a company based in Paris and New York"

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
            model="DistilBERT-TokenClassification",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load DistilBERT model for token classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DistilBERT model instance.
        """

        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DistilBertForTokenClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for DistilBERT token classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            # Ensure tokenizer is initialized
            self.load_model(dtype_override=dtype_override)

        # Data preprocessing
        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for token classification.

        Args:
            co_out: Model output
            framework_model: Framework model with config (needed for id2label mapping)
        """
        inputs = self.load_inputs()
        predicted_token_class_ids = co_out[0].argmax(-1)
        predicted_token_class_ids = torch.masked_select(
            predicted_token_class_ids, (inputs["attention_mask"][0] == 1)
        )

        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_tokens_classes = [
                framework_model.config.id2label[t.item()]
                for t in predicted_token_class_ids
            ]
            print(f"Context: {self.sample_text}")
            print(f"Answer: {predicted_tokens_classes}")
        else:
            print(f"Context: {self.sample_text}")
            print(f"Predicted token class IDs: {predicted_token_class_ids.tolist()}")
