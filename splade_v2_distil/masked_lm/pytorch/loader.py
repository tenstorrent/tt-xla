# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SPLADE v2 Distil model loader implementation for masked language modeling."""

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


class ModelVariant(StrEnum):
    """Available SPLADE v2 Distil model variants."""

    SPLADE_V2_DISTIL = "V2_Distil"


class ModelLoader(ForgeModel):
    """SPLADE v2 Distil model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.SPLADE_V2_DISTIL: LLMModelConfig(
            pretrained_model_name="naver/splade_v2_distil",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SPLADE_V2_DISTIL

    sample_text = "Semantic search using sparse retrieval with learned term weights."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""

        return ModelInfo(
            model="SPLADE v2 Distil",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SPLADE v2 Distil model instance."""

        from transformers import AutoModelForMaskedLM

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        self.model = AutoModelForMaskedLM.from_pretrained(
            self._model_name, **model_kwargs
        )
        self.model.eval()

        return self.model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        return inputs
