# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT model loader implementation for token classification.
"""

import torch
from transformers import AutoTokenizer, BertForTokenClassification
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
    """Available BERT model variants for token classification."""

    DBMDZ_BERT_LARGE_CASED_FINETUNED_CONLL03_ENGLISH = (
        "dbmdz/bert-large-cased-finetuned-conll03-english"
    )
    DSLIM_BERT_BASE_NER = "dslim/bert-base-NER"
    DAVLAN_BERT_BASE_MULTILINGUAL_CASED_NER_HRL = (
        "Davlan/bert-base-multilingual-cased-ner-hrl"
    )
    HATMIMOHA_ARABIC_NER = "hatmimoha/arabic-ner"
    CAMEL_LAB_BERT_BASE_ARABIC_CAMELBERT_MSA_NER = (
        "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
    )
    P208P2002_ZH_WIKI_PUNCTUATION_RESTORE = "p208p2002/zh-wiki-punctuation-restore"


class ModelLoader(ForgeModel):
    """BERT model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DBMDZ_BERT_LARGE_CASED_FINETUNED_CONLL03_ENGLISH: LLMModelConfig(
            pretrained_model_name="dbmdz/bert-large-cased-finetuned-conll03-english",
            max_length=128,
        ),
        ModelVariant.DSLIM_BERT_BASE_NER: LLMModelConfig(
            pretrained_model_name="dslim/bert-base-NER",
            max_length=128,
        ),
        ModelVariant.DAVLAN_BERT_BASE_MULTILINGUAL_CASED_NER_HRL: LLMModelConfig(
            pretrained_model_name="Davlan/bert-base-multilingual-cased-ner-hrl",
            max_length=128,
        ),
        ModelVariant.HATMIMOHA_ARABIC_NER: LLMModelConfig(
            pretrained_model_name="hatmimoha/arabic-ner",
            max_length=128,
        ),
        ModelVariant.CAMEL_LAB_BERT_BASE_ARABIC_CAMELBERT_MSA_NER: LLMModelConfig(
            pretrained_model_name="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
            max_length=128,
        ),
        ModelVariant.P208P2002_ZH_WIKI_PUNCTUATION_RESTORE: LLMModelConfig(
            pretrained_model_name="p208p2002/zh-wiki-punctuation-restore",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DBMDZ_BERT_LARGE_CASED_FINETUNED_CONLL03_ENGLISH

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
        if self._variant in (
            ModelVariant.HATMIMOHA_ARABIC_NER,
            ModelVariant.CAMEL_LAB_BERT_BASE_ARABIC_CAMELBERT_MSA_NER,
        ):
            self.sample_text = (
                "إمارة أبوظبي هي إحدى إمارات دولة الإمارات العربية المتحدة السبع"
            )
        elif self._variant == ModelVariant.P208P2002_ZH_WIKI_PUNCTUATION_RESTORE:
            self.sample_text = "在一般情况下句子的结尾要用句号来表示停顿"
        else:
            self.sample_text = "HuggingFace is a company based in Paris and New York"
        self.max_length = self._variant_config.max_length
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

        group = ModelGroup.GENERALITY
        if variant_name in (
            ModelVariant.DAVLAN_BERT_BASE_MULTILINGUAL_CASED_NER_HRL,
            ModelVariant.DSLIM_BERT_BASE_NER,
            ModelVariant.DSLIM_BERT_BASE_NER_UNCASED,
            ModelVariant.HATMIMOHA_ARABIC_NER,
            ModelVariant.CAMEL_LAB_BERT_BASE_ARABIC_CAMELBERT_MSA_NER,
            ModelVariant.P208P2002_ZH_WIKI_PUNCTUATION_RESTORE,
        ):
            group = ModelGroup.VULCAN

        return ModelInfo(
            model="BERT",
            variant=variant_name,
            group=group,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BERT model for token classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BERT model instance.
        """

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertForTokenClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for BERT token classification.

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

    def decode_output(self, co_out):
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
        predicted_tokens_classes = [
            self.model.config.id2label[t.item()] for t in predicted_token_class_ids
        ]

        print(f"Context: {self.sample_text}")
        print(f"Answer: {predicted_tokens_classes}")
