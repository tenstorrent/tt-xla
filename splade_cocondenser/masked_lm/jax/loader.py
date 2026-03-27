# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""SPLADE CoCondenser model loader implementation for masked language modeling."""

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
    """Available SPLADE CoCondenser model variants."""

    ENSEMBLEDISTIL = "EnsembleDistil"


class ModelLoader(ForgeModel):
    """SPLADE CoCondenser model loader implementation for masked language modeling."""

    _VARIANTS = {
        ModelVariant.ENSEMBLEDISTIL: LLMModelConfig(
            pretrained_model_name="naver/splade-cocondenser-ensembledistil",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ENSEMBLEDISTIL

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
            model="SPLADE CoCondenser",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_MASKED_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""

        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SPLADE CoCondenser model instance."""

        from transformers import FlaxAutoModelForMaskedLM

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        model = FlaxAutoModelForMaskedLM.from_pretrained(
            self._model_name, from_pt=True, **model_kwargs
        )

        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
        )

        return inputs
