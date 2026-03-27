# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SPLADE Efficient VI-BT-large-doc model loader implementation for masked language modeling.
"""

from transformers import AutoModelForMaskedLM, AutoTokenizer
from typing import Optional

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
    """Available SPLADE Efficient VI-BT-large model variants."""

    EFFICIENT_SPLADE_VI_BT_LARGE_DOC = "efficient-splade-VI-BT-large-doc"


class ModelLoader(ForgeModel):
    """SPLADE Efficient VI-BT-large-doc model loader for masked language modeling."""

    _VARIANTS = {
        ModelVariant.EFFICIENT_SPLADE_VI_BT_LARGE_DOC: LLMModelConfig(
            pretrained_model_name="naver/efficient-splade-VI-BT-large-doc",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.EFFICIENT_SPLADE_VI_BT_LARGE_DOC

    sample_text = "Semantic search using sparse retrieval with learned term weights."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""

        return ModelInfo(
            model="SPLADE Efficient VI-BT-large",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load SPLADE Efficient VI-BT-large-doc model from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The model instance.
        """

        if self._tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(self._model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""

        if self._tokenizer is None:
            self._load_tokenizer()

        inputs = self._tokenizer(
            self.sample_text,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs
