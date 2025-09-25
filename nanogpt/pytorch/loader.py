# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NanoGPT model loader implementation
"""

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from ...base import ForgeModel

from transformers import AutoModel, AutoTokenizer


class ModelVariant(StrEnum):
    """Available NanoGPT model variants."""

    FINANCIAL_SUPPORT_NANOGPT = "FinancialSupport/NanoGPT"


class ModelLoader(ForgeModel):
    """NanoGPT model loader implementation."""

    _VARIANTS = {
        ModelVariant.FINANCIAL_SUPPORT_NANOGPT: ModelConfig(
            pretrained_model_name="FinancialSupport/NanoGPT",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.FINANCIAL_SUPPORT_NANOGPT

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
            model="nanogpt",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    """Loads NanoGPT model and sample input."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "FinancialSupport/NanoGPT"

    def load_model(self, dtype_override=None):
        """Load pretrained NanoGPT model."""
        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name

        model = AutoModel.from_pretrained(
            pretrained_model_name,
            ignore_mismatched_sizes=True,
            use_cache=False,
        )
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for NanoGPT model"""

        # Get the pretrained model name from the instance's variant config
        pretrained_model_name = self._variant_config.pretrained_model_name
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Input prompt
        input_prompt = "The financial market showed signs of volatility"

        # Tokenize input
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            max_length=150,
            padding=True,
            truncation=True,
        )

        return inputs
