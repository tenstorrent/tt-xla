# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BLEURT model loader implementation for sequence classification.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
    """Available BLEURT model variants for sequence classification."""

    BASE_128 = "base-128"


class ModelLoader(ForgeModel):
    """BLEURT model loader implementation for sequence classification."""

    _VARIANTS = {
        ModelVariant.BASE_128: ModelConfig(
            pretrained_model_name="Elron/bleurt-base-128",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE_128

    # Sample reference-candidate pairs for testing
    sample_pairs = [
        ("hello world", "hi universe"),
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="BLEURT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name,
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load BLEURT model for sequence classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The BLEURT model instance.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name, **model_kwargs
        )
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample inputs for BLEURT evaluation.

        Args:
            dtype_override: Optional torch.dtype to override input tensor dtype.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        references = [pair[0] for pair in self.sample_pairs]
        candidates = [pair[1] for pair in self.sample_pairs]

        inputs = self.tokenizer(
            references,
            candidates,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128,
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
