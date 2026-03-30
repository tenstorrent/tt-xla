# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
NVIDIA Cosmos Tokenizer model loader implementation for continuous video tokenization.
"""

import torch
from huggingface_hub import hf_hub_download
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
    """Available Cosmos Tokenizer model variants."""

    CV8X8X8_720P = "CV8x8x8-720p"


class ModelLoader(ForgeModel):
    """NVIDIA Cosmos Tokenizer model loader for continuous video tokenization."""

    _VARIANTS = {
        ModelVariant.CV8X8X8_720P: ModelConfig(
            pretrained_model_name="nvidia/Cosmos-Tokenize1-CV8x8x8-720p",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CV8X8X8_720P

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Cosmos-Tokenizer",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Cosmos Tokenizer autoencoder JIT model."""
        pretrained_model_name = self._variant_config.pretrained_model_name

        jit_path = hf_hub_download(
            repo_id=pretrained_model_name, filename="autoencoder.jit"
        )
        model = torch.jit.load(jit_path)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return synthetic video inputs for the Cosmos Tokenizer.

        The model expects input shape [B, C, T, H, W] with T as a multiple of 8 + 1.
        """
        # 9 frames (8+1 for causal temporal compression), 256x256 spatial
        inputs = torch.randn(batch_size, 3, 9, 256, 256, dtype=torch.bfloat16)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
