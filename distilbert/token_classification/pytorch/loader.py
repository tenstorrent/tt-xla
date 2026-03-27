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
    OPENMED_NER_ONCOLOGY_DETECT = "OpenMed_NER_OncologyDetect_TinyMed_66M"
    OPENMED_NER_GENOMIC_DETECT = "OpenMed_NER_GenomicDetect_TinyMed_135M"
    D4DATA_BIOMEDICAL_NER_ALL = "d4data/biomedical-ner-all"


class ModelLoader(ForgeModel):
    """DistilBERT model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL: LLMModelConfig(
            pretrained_model_name="Davlan/distilbert-base-multilingual-cased-ner-hrl",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_ONCOLOGY_DETECT: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-OncologyDetect-TinyMed-66M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_GENOMIC_DETECT: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomicDetect-TinyMed-135M",
            max_length=128,
        ),
        ModelVariant.D4DATA_BIOMEDICAL_NER_ALL: LLMModelConfig(
            pretrained_model_name="d4data/biomedical-ner-all",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL

    _SAMPLE_TEXTS = {
        ModelVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL: (
            "HuggingFace is a company based in Paris and New York"
        ),
        ModelVariant.OPENMED_NER_ONCOLOGY_DETECT: (
            "Mutations in KRAS gene drive oncogenic transformation."
        ),
        ModelVariant.OPENMED_NER_GENOMIC_DETECT: (
            "The BRCA2 gene is associated with hereditary breast cancer."
        ),
        ModelVariant.D4DATA_BIOMEDICAL_NER_ALL: (
            "The patient reported no recurrence of palpitations at follow-up 6 months after the ablation."
        ),
    }

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
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self.sample_text = self._SAMPLE_TEXTS[self._variant]

    _VARIANT_GROUPS = {
        ModelVariant.DAVLAN_DISTILBERT_BASE_MULTILINGUAL_CASED_NER_HRL: ModelGroup.GENERALITY,
        ModelVariant.OPENMED_NER_ONCOLOGY_DETECT: ModelGroup.VULCAN,
        ModelVariant.OPENMED_NER_GENOMIC_DETECT: ModelGroup.VULCAN,
        ModelVariant.D4DATA_BIOMEDICAL_NER_ALL: ModelGroup.VULCAN,
    }

    @classmethod
    def _get_model_info(cls, variant=None):
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DistilBERT",
            variant=variant,
            group=cls._VARIANT_GROUPS.get(variant, ModelGroup.GENERALITY),
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
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
        model_kwargs |= kwargs

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
