# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
RoBERTa model loader implementation for feature extraction.
"""
import torch
from transformers import AutoModel, AutoTokenizer
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
    """Available RoBERTa model variants for feature extraction."""

    TINY_RANDOM = "peft-internal-testing/tiny-random-RobertaModel"


class ModelLoader(ForgeModel):
    """RoBERTa model loader implementation for feature extraction."""

    _VARIANTS = {
        ModelVariant.TINY_RANDOM: LLMModelConfig(
            pretrained_model_name="peft-internal-testing/tiny-random-RobertaModel",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.model = None
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="RoBERTa",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant."""
        if self.tokenizer is None:
            model_name = self._variant_config.pretrained_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load RoBERTa model for feature extraction from Hugging Face."""
        if self.tokenizer is None:
            self._load_tokenizer()

        model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(model_name, **model_kwargs)
        model.eval()

        self.model = model
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model."""
        if self.tokenizer is None:
            self._load_tokenizer()

        sample_text = "The capital of France is Paris."
        max_length = self._variant_config.max_length

        inputs = self.tokenizer(
            sample_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, outputs):
        """Decode the model output for feature extraction."""
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        elif hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output to extract a differentiable tensor."""
        tensors = []

        if hasattr(fwd_output, "last_hidden_state"):
            tensors.append(fwd_output.last_hidden_state.flatten())

        if tensors:
            return torch.cat(tensors, dim=0)
        return fwd_output
