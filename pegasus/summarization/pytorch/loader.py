# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pegasus PyTorch model loader implementation for text summarization.
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
    """Available Pegasus model variants."""

    CNN_DAILYMAIL = "Cnn Dailymail"


class ModelLoader(ForgeModel):
    """Pegasus PyTorch model loader implementation for text summarization."""

    _VARIANTS = {
        ModelVariant.CNN_DAILYMAIL: LLMModelConfig(
            pretrained_model_name="google/pegasus-cnn_dailymail",
            max_length=1024,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CNN_DAILYMAIL

    sample_text = "My friends are cool but they eat too many carbs."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer = None
        self._cached_model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="Pegasus",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_SUMMARIZATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Pegasus model instance."""
        from transformers import PegasusForConditionalGeneration

        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = PegasusForConditionalGeneration.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Pegasus model."""
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        # Pegasus requires decoder input ids as an input for seq2seq
        decoder_start_token_tensor = torch.tensor(
            self._cached_model.generation_config.decoder_start_token_id,
            dtype=torch.long,
        )
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_tensor
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
