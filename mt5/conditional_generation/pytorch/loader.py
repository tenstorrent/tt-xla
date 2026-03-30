# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""MT5 PyTorch model loader implementation for conditional generation (SMILES to IUPAC)."""

from typing import Optional

import torch
from transformers import AutoTokenizer, MT5Config, MT5ForConditionalGeneration

from ....base import ForgeModel
from ....config import (
    Framework,
    LLMModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available MT5 conditional generation model variants."""

    SMILES2IUPAC_CANONICAL_SMALL = "SMILES2IUPAC_canonical_small"


class ModelLoader(ForgeModel):
    """MT5 PyTorch model loader implementation for conditional generation."""

    _VARIANTS = {
        ModelVariant.SMILES2IUPAC_CANONICAL_SMALL: LLMModelConfig(
            pretrained_model_name="knowledgator/SMILES2IUPAC-canonical-small",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SMILES2IUPAC_CANONICAL_SMALL

    sample_text = "CCO"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._cached_model = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="mT5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the tokenizer's default dtype.

        Returns:
            The loaded tokenizer instance
        """
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the MT5 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The MT5 model instance for conditional generation.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {"use_cache": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = MT5ForConditionalGeneration.from_pretrained(
            self._model_name, **model_kwargs
        )
        model.eval()
        self._cached_model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the MT5 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="pt",
        )

        config = MT5Config.from_pretrained(self._model_name)
        decoder_start_token_id = config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
