# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
PersonaPlex 7B MLX 4-bit speech-text dialogue model loader implementation.
"""

import torch
from huggingface_hub import hf_hub_download
from moshi.models import loaders
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
    """Available PersonaPlex 7B MLX 4-bit model variants."""

    PERSONAPLEX_7B_MLX_4BIT = "PersonaPlex 7B MLX 4bit"


class ModelLoader(ForgeModel):
    """PersonaPlex 7B MLX 4-bit speech-text dialogue model loader implementation."""

    _VARIANTS = {
        ModelVariant.PERSONAPLEX_7B_MLX_4BIT: ModelConfig(
            pretrained_model_name="aufklarer/PersonaPlex-7B-MLX-4bit",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.PERSONAPLEX_7B_MLX_4BIT

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="PersonaPlex 7B MLX 4-bit",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the PersonaPlex LM model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        weight_path = hf_hub_download(pretrained_model_name, loaders.MOSHI_NAME)

        dtype = dtype_override if dtype_override is not None else torch.float32
        model = loaders.get_moshi_lm(weight_path, device="cpu", dtype=dtype)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample code inputs for the PersonaPlex model.

        The model expects discrete audio codes of shape [B, K, T] where
        K=17 codebooks (1 text + 8 user audio + 8 agent audio) and T is time steps.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name
        weight_path = hf_hub_download(pretrained_model_name, loaders.MOSHI_NAME)
        model = loaders.get_moshi_lm(weight_path, device="cpu", dtype=torch.float32)

        codes = torch.randint(0, model.card, (1, model.num_codebooks, 10))

        del model
        return codes
