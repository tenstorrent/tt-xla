# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
BERT model loader implementation for masked language modeling.
"""

from transformers import AutoTokenizer, FlaxBertForMaskedLM
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available BERT model variants."""

    BASE = "Base"
    BASE_CASED = "Base (Cased)"
    BASE_CHINESE = "Base (Chinese)"
    LARGE = "Large"
    MULTILINGUAL_BASE = "Multilingual Base (Uncased)"


class ModelLoader(ForgeModel):
    """BERT model loader implementation for masked language modeling."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="google-bert/bert-base-uncased",
        ),
        ModelVariant.BASE_CASED: LLMModelConfig(
            pretrained_model_name="google-bert/bert-base-cased",
        ),
        ModelVariant.BASE_CHINESE: LLMModelConfig(
            pretrained_model_name="google-bert/bert-base-chinese",
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="google-bert/bert-large-uncased",
        ),
        ModelVariant.MULTILINGUAL_BASE: LLMModelConfig(
            pretrained_model_name="google-bert/bert-base-multilingual-uncased",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    _SAMPLE_TEXTS = {
        ModelVariant.BASE_CHINESE: "\u6211\u7231[\u004d\u0041\u0053\u004b]\u56fd\u3002",
    }

    _DEFAULT_SAMPLE_TEXT = "The capital of France is [MASK]."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None

    @property
    def sample_text(self):
        return self._SAMPLE_TEXTS.get(self._variant, self._DEFAULT_SAMPLE_TEXT)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        variant_groups = {
            ModelVariant.BASE_CASED: ModelGroup.VULCAN,
            ModelVariant.BASE_CHINESE: ModelGroup.VULCAN,
            ModelVariant.MULTILINGUAL_BASE: ModelGroup.VULCAN,
        }

        return ModelInfo(
            model="BERT",
            variant=variant,
            group=variant_groups.get(variant, ModelGroup.GENERALITY),
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load the tokenizer for the model.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            Tokenizer: The tokenizer for the model
        """

        # Initialize tokenizer with dtype_override if provided
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the BERT model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            model: The loaded model instance
        """

        pretrained_model_name = self._variant_config.pretrained_model_name

        # Ensure tokenizer is loaded
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        # Load the model
        model = FlaxBertForMaskedLM.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BERT model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        # Ensure tokenizer is initialized
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Create tokenized inputs for the masked language modeling task
        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
