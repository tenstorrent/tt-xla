# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ModernBERT model loader implementation for sequence classification.
"""

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
    """Available ModernBERT model variants for sequence classification."""

    FINEPDFS_EDU_CLASSIFIER_ENG_LATN = "finepdfs_edu_classifier_eng_Latn"


class ModelLoader(ForgeModel):
    """ModernBERT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.FINEPDFS_EDU_CLASSIFIER_ENG_LATN: LLMModelConfig(
            pretrained_model_name="HuggingFaceFW/finepdfs_edu_classifier_eng_Latn",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.FINEPDFS_EDU_CLASSIFIER_ENG_LATN

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = self._variant_config.pretrained_model_name
        self.max_length = self._variant_config.max_length
        self.sample_text = "Photosynthesis is the process by which green plants convert sunlight into chemical energy."
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses default.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ModernBERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load ModernBERT model for sequence classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The ModernBERT model instance.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, **model_kwargs
        )
        model.eval()
        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for ModernBERT sequence classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
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
        logits = co_out[0]
        score = logits.squeeze(-1).item()
        print(f"Educational quality score: {score:.2f}")
