# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Moshiko speech-text dialogue model loader implementation.
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
    """Available Moshiko model variants."""

    MOSHIKO_BF16 = "Moshiko BF16"
    MOSHIKO_Q8 = "Moshiko Q8"


class ModelLoader(ForgeModel):
    """Moshiko speech-text dialogue model loader implementation."""

    _VARIANTS = {
        ModelVariant.MOSHIKO_BF16: ModelConfig(
            pretrained_model_name="kyutai/moshiko-pytorch-bf16",
        ),
        ModelVariant.MOSHIKO_Q8: ModelConfig(
            pretrained_model_name="kyutai/moshiko-pytorch-q8",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MOSHIKO_BF16

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Moshiko",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Moshiko LM model instance."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        weight_path = hf_hub_download(pretrained_model_name, loaders.MOSHI_NAME)

        dtype = dtype_override if dtype_override is not None else torch.float32
        model = loaders.get_moshi_lm(weight_path, device="cpu", dtype=dtype)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None):
        """Load sample code inputs for the Moshiko model.

        The model expects discrete audio codes of shape [B, K, T] where
        K=17 codebooks (1 text + 8 user audio + 8 model audio) and T is time steps.
        """
        # Load model briefly to get num_codebooks and card
        pretrained_model_name = self._variant_config.pretrained_model_name
        weight_path = hf_hub_download(pretrained_model_name, loaders.MOSHI_NAME)
        model = loaders.get_moshi_lm(weight_path, device="cpu", dtype=torch.float32)

        # Generate synthetic discrete codes: [B=1, K=17, T=10]
        codes = torch.randint(0, model.card, (1, model.num_codebooks, 10))

        del model
        return codes
