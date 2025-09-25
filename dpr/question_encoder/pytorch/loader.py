# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
DPR Question Encoder model loader implementation
"""


from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
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
    """Available DPR Question Encoder model variants."""

    DPR_SINGLE_NQ_BASE = "facebook/dpr-question_encoder-single-nq-base"
    DPR_MULTISET_BASE = "facebook/dpr-question_encoder-multiset-base"


class ModelLoader(ForgeModel):
    """DPR Question Encoder model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DPR_SINGLE_NQ_BASE: LLMModelConfig(
            pretrained_model_name="facebook/dpr-question_encoder-single-nq-base",
            max_length=128,
        ),
        ModelVariant.DPR_MULTISET_BASE: LLMModelConfig(
            pretrained_model_name="facebook/dpr-question_encoder-multiset-base",
            max_length=128,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DPR_SINGLE_NQ_BASE

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
        self.text = "What is love ?"
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
            model="DPR-Question-Encoder",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the DPR Question Encoder model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DPR Question Encoder model instance.

        """

        # Initialize tokenizer
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = DPRQuestionEncoder.from_pretrained(self.model_name, **model_kwargs)
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the DPR Question Encoder model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            # Ensure tokenizer is initialized
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
