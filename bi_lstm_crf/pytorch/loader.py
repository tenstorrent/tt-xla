# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiLSTM-CRF model loader implementation
"""
import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .src.model_utils import (
    create_bi_lstm_crf_model,
    create_sample_input,
    get_vocab_mappings,
)


class ModelVariant(StrEnum):
    """Available BiLSTM-CRF model variants."""

    DEFAULT = "default"


class ModelLoader(ForgeModel):
    """BiLSTM-CRF model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="bi_lstm_crf_default",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEFAULT

    # Shared configuration parameters
    test_sentence = ["apple", "corporation", "is", "in", "georgia"]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

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
        return ModelInfo(
            model="bi_lstm_crf",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the BiLSTM-CRF model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The BiLSTM-CRF model instance.
        """
        # Create the BiLSTM-CRF model
        model = create_bi_lstm_crf_model()

        # Apply dtype conversion if specified
        if dtype_override is not None:
            model = model.to(dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None, test_sentence=None):
        """Load and return sample inputs for the BiLSTM-CRF model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            test_sentence: Optional list of words to use as test sentence.
                          If None, uses default test sentence.

        Returns:
            torch.Tensor: Input tensor that can be fed to the model.
        """
        # Use provided sentence or default
        sentence = test_sentence if test_sentence is not None else self.test_sentence

        # Create input tensor
        test_input = create_sample_input(sentence)

        # Apply dtype conversion if specified
        if dtype_override is not None:
            test_input = test_input.to(dtype_override)

        return test_input
