# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Vietnamese Punctuation, Capitalization & Denormalization model loader implementation.
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
    """Available Vietnamese Punc Cap Denorm model variants."""

    V1 = "V1"


class ModelLoader(ForgeModel):
    """Vietnamese Punc Cap Denorm model loader for text2text generation."""

    _VARIANTS = {
        ModelVariant.V1: LLMModelConfig(
            pretrained_model_name="tourmii/vietnamese-punc-cap-denorm-v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    sample_text = (
        "phát biểu nhậm chức chiều hai mươi tám tháng mười một ông vũ đại thắng"
        " cho biết việc được tín nhiệm bầu làm chủ tịch ủy ban nhân dân thành phố"
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
            model="Vietnamese_Punc_Cap_Denorm",
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
        """Load and return the mBART model instance."""
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
        """Load and return sample inputs for the model."""
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        decoder_start_token_id = self._model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]])
        inputs["decoder_input_ids"] = decoder_input_ids

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                    inputs[key] = value.to(dtype_override)

        return inputs
