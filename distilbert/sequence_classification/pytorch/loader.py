# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DistilBERT model loader implementation for sequence classification.
"""

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
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
    """Available DistilBERT model variants for sequence classification."""

    DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH = (
        "distilbert-base-uncased-finetuned-sst-2-english"
    )


class ModelLoader(ForgeModel):
    """DistilBERT model loader implementation for sequence classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH: LLMModelConfig(
            pretrained_model_name="distilbert-base-uncased-finetuned-sst-2-english",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DISTILBERT_BASE_UNCASED_FINETUNED_SST_2_ENGLISH

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
        self.review = "the movie was great!"

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
            model="DistilBERT-SequenceClassification",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load DistilBERT model for sequence classification from Hugging Face.

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

        model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for DistilBERT sequence classification.

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
            self.review,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out, framework_model=None):
        """Decode the model output for sequence classification.

        Args:
            co_out: Model output
            framework_model: Framework model with config (needed for id2label mapping)
        """
        predicted_class_id = co_out[0].argmax().item()
        if (
            framework_model
            and hasattr(framework_model, "config")
            and hasattr(framework_model.config, "id2label")
        ):
            predicted_category = framework_model.config.id2label[predicted_class_id]
            print(f"Predicted category: {predicted_category}")
        else:
            print(f"Predicted class ID: {predicted_class_id}")
