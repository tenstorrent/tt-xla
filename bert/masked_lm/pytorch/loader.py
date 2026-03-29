# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BERT model loader implementation for masked language modeling.
"""

from transformers import BertForMaskedLM, BertTokenizer, AutoConfig, AutoTokenizer
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

_SAMPLE_TEXTS = {
    "dbmdz/bert-base-german-uncased": "Die Hauptstadt von Deutschland ist [MASK].",
}


class ModelVariant(StrEnum):
    """Available BERT model variants for masked language modeling."""

    BERT_BASE_UNCASED = "Base_Uncased"
    BERT_BASE_CASED = "Base_Cased"
    BERT_LARGE_CASED = "Large_Cased"
    BERT_BASE_MULTILINGUAL_CASED = "Base_Multilingual_Cased"
    BIO_CLINICAL_BERT = "Bio_ClinicalBERT"
    BIOBERT_BASE_CASED_V1_1 = "BioBERT_Base_Cased_v1.1"
    BERT_LARGE_PORTUGUESE_CASED = "Large_Portuguese_Cased"
    LEGAL_BERT_BASE_UNCASED = "nlpaueb/legal-bert-base-uncased"
    RETROMAE_MSMARCO_DISTILL = "Shitao/RetroMAE_MSMARCO_distill"
    BERT_LARGE_UNCASED_WWM = "Large_Uncased_Whole_Word_Masking"


class ModelLoader(ForgeModel):
    """BERT model loader implementation for masked language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BERT_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="bert-base-uncased",
            max_length=128,
        ),
        ModelVariant.BERT_BASE_CASED: LLMModelConfig(
            pretrained_model_name="google-bert/bert-base-cased",
            max_length=128,
        ),
        ModelVariant.BERT_LARGE_CASED: LLMModelConfig(
            pretrained_model_name="google-bert/bert-large-cased",
            max_length=128,
        ),
        ModelVariant.BERT_BASE_MULTILINGUAL_CASED: LLMModelConfig(
            pretrained_model_name="google-bert/bert-base-multilingual-cased",
            max_length=128,
        ),
        ModelVariant.BIO_CLINICAL_BERT: LLMModelConfig(
            pretrained_model_name="emilyalsentzer/Bio_ClinicalBERT",
            max_length=128,
        ),
        ModelVariant.BIOBERT_BASE_CASED_V1_1: LLMModelConfig(
            pretrained_model_name="dmis-lab/biobert-base-cased-v1.1",
            max_length=128,
        ),
        ModelVariant.BERT_LARGE_PORTUGUESE_CASED: LLMModelConfig(
            pretrained_model_name="neuralmind/bert-large-portuguese-cased",
            max_length=128,
        ),
        ModelVariant.LEGAL_BERT_BASE_UNCASED: LLMModelConfig(
            pretrained_model_name="nlpaueb/legal-bert-base-uncased",
            max_length=128,
        ),
        ModelVariant.RETROMAE_MSMARCO_DISTILL: LLMModelConfig(
            pretrained_model_name="Shitao/RetroMAE_MSMARCO_distill",
            max_length=128,
        ),
        ModelVariant.BERT_LARGE_UNCASED_WWM: LLMModelConfig(
            pretrained_model_name="bert-large-uncased-whole-word-masking",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BERT_BASE_UNCASED

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
        self.sample_text = _SAMPLE_TEXTS.get(
            self._variant, "The capital of France is [MASK]."
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
        group = ModelGroup.GENERALITY
        if variant_name in (
            ModelVariant.BERT_BASE_CASED,
            ModelVariant.BERT_LARGE_CASED,
            ModelVariant.BERT_BASE_MULTILINGUAL_CASED,
            ModelVariant.BIO_CLINICAL_BERT,
            ModelVariant.BIOBERT_BASE_CASED_V1_1,
            ModelVariant.BERT_LARGE_PORTUGUESE_CASED,
            ModelVariant.LEGAL_BERT_BASE_UNCASED,
            ModelVariant.RETROMAE_MSMARCO_DISTILL,
            ModelVariant.BERT_LARGE_UNCASED_WWM,
        ):
            group = ModelGroup.VULCAN
        return ModelInfo(
            model="BERT",
            variant=variant_name,
            group=group,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BERT model for masked language modeling from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BERT model instance.
        """

        # Initialize tokenizer
        if self._variant == ModelVariant.TOHOKU_NLP_BERT_BASE_JAPANESE_CHAR_V2:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = BertForMaskedLM.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for BERT masked language modeling.

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
        """Decode the model output for masked language modeling."""
        inputs = self.load_inputs()
        logits = co_out[0]
        mask_token_index = (inputs["input_ids"] == self.tokenizer.mask_token_id)[
            0
        ].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
        predicted_token = self.tokenizer.decode(predicted_token_id)
        print("The predicted token for the [MASK] is:", predicted_token)

    def load_config(self):
        """Load and return the configuration for the Bert model variant.

        Returns:
            The configuration object for the Bert model.
        """
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.config
