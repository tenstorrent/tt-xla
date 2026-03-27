# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ByT5 model loader implementation for grapheme-to-phoneme conversion.
"""

import torch
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
    """Available ByT5 G2P model variants."""

    MULTILINGUAL_SMALL_100 = "Multilingual_Small_100"


class ModelLoader(ForgeModel):
    """ByT5 model loader implementation for grapheme-to-phoneme conversion."""

    _VARIANTS = {
        ModelVariant.MULTILINGUAL_SMALL_100: LLMModelConfig(
            pretrained_model_name="charsiu/g2p_multilingual_byT5_small_100",
            max_length=50,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MULTILINGUAL_SMALL_100

    _SAMPLE_TEXTS = {
        ModelVariant.MULTILINGUAL_SMALL_100: "<eng-us>: hello",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="ByT5_G2P",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Note: The ByT5 tokenizer must be loaded from google/byt5-small
        as the model repo does not include tokenizer files.
        """
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ByT5 G2P model instance for this instance's variant."""
        from transformers import T5ForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = T5ForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ByT5 G2P model."""
        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self._SAMPLE_TEXTS[self._variant],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
