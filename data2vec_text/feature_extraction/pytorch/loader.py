# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Data2VecText model loader implementation for feature extraction (PyTorch).
"""

import torch
from transformers import AutoTokenizer, Data2VecTextModel
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
    """Available Data2VecText feature extraction model variants."""

    TINY_RANDOM = "Tiny_Random"


class ModelLoader(ForgeModel):
    """Data2VecText model loader implementation for feature extraction (PyTorch)."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-Data2VecTextModel",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    sample_text = "The capital of France is Paris."

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Data2VecText",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Data2VecText model instance for feature extraction.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Data2VecText model instance for feature extraction.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Data2VecTextModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Data2VecText model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            self.sample_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs, inputs=None):
        """Extract embeddings from model outputs.

        Args:
            outputs: Model output from a forward pass
            inputs: Optional input tensors used to generate the outputs

        Returns:
            torch.Tensor: CLS token embedding
        """
        if isinstance(outputs, (tuple, list)):
            last_hidden_state = outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        else:
            last_hidden_state = outputs

        # Use [CLS] token embedding as the sentence representation
        cls_embedding = last_hidden_state[:, 0, :]
        return cls_embedding
