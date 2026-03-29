# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
OpenSearch Neural Sparse Encoding model loader implementation for masked language modeling.
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available OpenSearch Neural Sparse Encoding model variants."""

    DOC_V3_DISTILL = "Doc_v3_Distill"


class ModelLoader(ForgeModel):
    """OpenSearch Neural Sparse Encoding model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.DOC_V3_DISTILL: ModelConfig(
            pretrained_model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DOC_V3_DISTILL

    sample_text = "Semantic search using sparse retrieval with learned term weights."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""

        return ModelInfo(
            model="OpenSearch Neural Sparse Encoding",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the OpenSearch Neural Sparse Encoding model instance."""

        from transformers import AutoModelForMaskedLM

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForMaskedLM.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""

        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        return inputs
