# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""ModernBERT Embed Base Finetune 8192 model loader implementation for embedding generation."""

import torch
from typing import Optional

from transformers import AutoModel, AutoTokenizer

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
    """Available ModernBERT Embed Base Finetune 8192 model variants."""

    MODERNBERT_EMBED_BASE_FINETUNE_8192 = "modernbert-embed-base_finetune_8192"


class ModelLoader(ForgeModel):
    """ModernBERT Embed Base Finetune 8192 model loader for embedding generation."""

    _VARIANTS = {
        ModelVariant.MODERNBERT_EMBED_BASE_FINETUNE_8192: ModelConfig(
            pretrained_model_name="freelawproject/modernbert-embed-base_finetune_8192",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MODERNBERT_EMBED_BASE_FINETUNE_8192

    sample_sentences = [
        "What determines Medicaid reimbursement eligibility for restorative therapy in New York?"
    ]

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""

        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting."""

        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ModernBERT Embed Base Finetune 8192",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_EMBED_GEN,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )

        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load ModernBERT Embed Base Finetune 8192 model for embedding generation."""

        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {"return_dict": False}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for embedding generation."""

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        if dtype_override is not None:
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dtype == torch.float32:
                        inputs[key] = value.to(dtype_override)

        return inputs
