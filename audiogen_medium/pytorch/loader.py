# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AudioGen Medium model loader implementation for text-to-audio generation.
"""

import torch
from audiocraft.models import AudioGen

from ...base import ForgeModel
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)


class ModelLoader(ForgeModel):
    """AudioGen Medium model loader implementation."""

    def __init__(self, variant=None):
        super().__init__(variant)
        self.model_name = "facebook/audiogen-medium"
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="AudioGen Medium",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the AudioGen Medium model instance."""
        self.model = AudioGen.get_pretrained(self.model_name)
        self.model.set_generation_params(duration=5)
        return self.model

    def load_inputs(self, batch_size=1):
        """Load and return sample text descriptions for audio generation.

        Args:
            batch_size: Number of text descriptions to generate audio for.

        Returns:
            list: Text descriptions for audio generation.
        """
        descriptions = [
            "dog barking in a park",
            "rain falling on a tin roof",
        ]

        # Adjust to match requested batch size
        if batch_size <= len(descriptions):
            return descriptions[:batch_size]
        else:
            repeated = (descriptions * ((batch_size // len(descriptions)) + 1))[
                :batch_size
            ]
            return repeated
