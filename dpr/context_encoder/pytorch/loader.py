# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DPR model loader implementation
"""

from transformers import DPRContextEncoderTokenizer, DPRContextEncoder
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
    """Available DPR Context Encoder model variants."""

    DPR_SINGLE_NQ_BASE = "facebook/dpr-ctx_encoder-single-nq-base"
    DPR_MULTISET_BASE = "facebook/dpr-ctx_encoder-multiset-base"


class ModelLoader(ForgeModel):
    """DPR Context Encoder model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DPR_SINGLE_NQ_BASE: LLMModelConfig(
            pretrained_model_name="facebook/dpr-ctx_encoder-single-nq-base",
            max_length=128,
        ),
        ModelVariant.DPR_MULTISET_BASE: LLMModelConfig(
            pretrained_model_name="facebook/dpr-ctx_encoder-multiset-base",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DPR_SINGLE_NQ_BASE

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
            model="DPR-Context-Encoder",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.text = "Hello, is my dog cute?"
        self.max_length = 128
        self.tokenizer = None

    def load_model(self, dtype_override=None):
        """Load a DPR Context Encoder model from Hugging Face."""

        # Initialize tokenizer first with default or overridden dtype
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            self.model_name, **tokenizer_kwargs
        )

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DPRContextEncoder.from_pretrained(self.model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self):
        """Generate sample inputs for DPR model."""

        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self.load_model()  # This will initialize the tokenizer

        # Create tokenized inputs
        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
