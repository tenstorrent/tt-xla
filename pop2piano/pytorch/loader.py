# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Pop2Piano model loader implementation
"""
import torch
import numpy as np
from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Pop2Piano model loader implementation for audio-to-MIDI piano cover generation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        self.model_name = "sweetcocoa/pop2piano"
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name=None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="Pop2Piano",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, **kwargs):
        """Load and return the Pop2Piano model instance.

        Returns:
            torch.nn.Module: The Pop2Piano model instance.
        """
        self.processor = Pop2PianoProcessor.from_pretrained(self.model_name, **kwargs)
        self.model = Pop2PianoForConditionalGeneration.from_pretrained(
            self.model_name, **kwargs
        )
        self.model.eval()
        return self.model

    def load_inputs(self, **kwargs):
        """Load and return sample inputs for the Pop2Piano model.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.processor is None:
            self.load_model()

        # Generate a synthetic audio waveform (sine wave) as sample input
        sr = 22050
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t)

        inputs = self.processor(audio=audio, sampling_rate=sr, return_tensors="pt")

        # Add decoder_input_ids for the encoder-decoder model
        decoder_start_token_id = self.model.generation_config.decoder_start_token_id
        decoder_input_ids = (
            torch.ones((1, 1), dtype=torch.long) * decoder_start_token_id
        )
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
