# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ESMFold (facebook/esmfold_v1) model loader implementation for protein structure prediction.
"""

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available ESMFold model variants."""

    ESMFOLD_V1 = "facebook/esmfold_v1"


class ModelLoader(ForgeModel):
    """ESMFold model loader implementation for protein structure prediction."""

    _VARIANTS = {
        ModelVariant.ESMFOLD_V1: ModelConfig(
            pretrained_model_name="facebook/esmfold_v1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ESMFOLD_V1

    # Short protein sequence for testing
    sample_sequence = "MGSSHHHHHHSSGLVPRGSHM"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ESMFold",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = EsmForProteinFolding.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self._load_tokenizer()

        inputs = self.tokenizer(
            self.sample_sequence,
            return_tensors="pt",
            add_special_tokens=False,
        )

        return inputs
