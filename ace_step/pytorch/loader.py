# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ACE-Step 1.5 model loader implementation for music generation tasks.
"""
import torch

from ...base import ForgeModel
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)


class ModelLoader(ForgeModel):
    """ACE-Step 1.5 model loader implementation for music generation."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = "ACE-Step/Ace-Step1.5"
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="ACE-Step 1.5",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **kwargs,
        )
        self.model.eval()
        return self.model

    def load_inputs(self, batch_size=1):
        # ACE-Step DiT expects latent audio representations
        # Model config: hidden_size=2048, in_channels=192, patch_size=2
        # Generate synthetic noise latents matching the expected input shape
        latents = torch.randn(batch_size, 192, 64, 64)
        timesteps = torch.randint(0, 1000, (batch_size,))
        # Conditioning embedding (text encoder output)
        encoder_hidden_states = torch.randn(batch_size, 77, 2048)
        return latents, timesteps, encoder_hidden_states
