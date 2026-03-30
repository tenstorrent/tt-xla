# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MBart model loader implementation for Vietnamese text correction.
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
    """Available MBart model variants for text correction."""

    BMD1905_VIETNAMESE_CORRECTION_V2 = "bmd1905/vietnamese-correction-v2"


class ModelLoader(ForgeModel):
    """MBart model loader implementation for Vietnamese text correction."""

    _VARIANTS = {
        ModelVariant.BMD1905_VIETNAMESE_CORRECTION_V2: LLMModelConfig(
            pretrained_model_name="bmd1905/vietnamese-correction-v2",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BMD1905_VIETNAMESE_CORRECTION_V2

    sample_text = (
        "côn viec kin doanh thì rất kho khan nên toi quyết dinh chuyển sang nghề khac"
    )

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self._tokenizer = None
        self._model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="MBart",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MBart model instance for this instance's variant."""
        from transformers import MBartForConditionalGeneration

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MBartForConditionalGeneration.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MBart text correction model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        if self._model is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            truncation=True,
            return_tensors="pt",
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
