# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenMed model loader implementation for token classification.
"""

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
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
    """Available OpenMed model variants for token classification."""

    OPENMED_NER_ANATOMY_DETECT_PUBMED_V2_109M = (
        "OpenMed/OpenMed-NER-AnatomyDetect-PubMed-v2-109M"
    )
    OPENMED_NER_DISEASE_DETECT_MODERNMED_149M = (
        "OpenMed/OpenMed-NER-DiseaseDetect-ModernMed-149M"
    )
    OPENMED_NER_GENOMIC_DETECT_TINYMED_65M = (
        "OpenMed/OpenMed-NER-GenomicDetect-TinyMed-65M"
    )
    OPENMED_PII_FRENCH_SNOWFLAKEMED_LARGE_568M_V1 = (
        "OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1"
    )
    OPENMED_PII_MODERNMED_BASE_149M_V1 = "OpenMed/OpenMed-PII-ModernMed-Base-149M-v1"


class ModelLoader(ForgeModel):
    """OpenMed model loader implementation for token classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.OPENMED_NER_ANATOMY_DETECT_PUBMED_V2_109M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-AnatomyDetect-PubMed-v2-109M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_DISEASE_DETECT_MODERNMED_149M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-DiseaseDetect-ModernMed-149M",
            max_length=128,
        ),
        ModelVariant.OPENMED_NER_GENOMIC_DETECT_TINYMED_65M: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-NER-GenomicDetect-TinyMed-65M",
            max_length=128,
        ),
        ModelVariant.OPENMED_PII_FRENCH_SNOWFLAKEMED_LARGE_568M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-French-SnowflakeMed-Large-568M-v1",
            max_length=128,
        ),
        ModelVariant.OPENMED_PII_MODERNMED_BASE_149M_V1: LLMModelConfig(
            pretrained_model_name="OpenMed/OpenMed-PII-ModernMed-Base-149M-v1",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.OPENMED_NER_ANATOMY_DETECT_PUBMED_V2_109M

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
        if self._variant == ModelVariant.OPENMED_NER_DISEASE_DETECT_MODERNMED_149M:
            self.sample_text = (
                "The patient was diagnosed with diabetes mellitus type 2."
            )
        elif self._variant == ModelVariant.OPENMED_NER_GENOMIC_DETECT_TINYMED_65M:
            self.sample_text = (
                "The BRCA2 gene is associated with hereditary breast cancer."
            )
        elif (
            self._variant == ModelVariant.OPENMED_PII_FRENCH_SNOWFLAKEMED_LARGE_568M_V1
        ):
            self.sample_text = "Patient Jean Martin, né le 15/03/1985, habite au 12 rue de la Paix, Paris."
        elif self._variant == ModelVariant.OPENMED_PII_MODERNMED_BASE_149M_V1:
            self.sample_text = "Dr. Sarah Johnson (SSN: 123-45-6789) can be reached at sarah.johnson@hospital.org or 555-123-4567."
        else:
            self.sample_text = (
                "The patient complained of pain in the left ventricle region."
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
            model="OpenMed",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load OpenMed model for token classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The model instance.
        """

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for OpenMed token classification.

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
