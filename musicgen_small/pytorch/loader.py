# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Musicgen-small model loader implementation
"""
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel


class ModelLoader(ForgeModel):
    """Musicgen-small model loader implementation."""

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Configuration parameters
        self.model_name = "facebook/musicgen-small"
        self.processor = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="musicgen_small",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load and return the Musicgen-small model instance with default settings.

        Returns:
            torch.nn.Module: The Musicgen-small model instance.

        """
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = MusicgenForConditionalGeneration.from_pretrained(self.model_name)
        return self.model

    def load_inputs(self, batch_size=1):
        """Load and return sample inputs for the Musicgen-small model with default settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        # Ensure processor is initialized
        if self.processor is None:
            self.load_model()

        inputs = self.processor(
            text=[
                "80s pop track with bassy drums and synth",
                "90s rock song with loud guitars and heavy drums",
            ],
            padding=True,
            return_tensors="pt",
        )

        # If batch_size is different from 2, adjust using repeat_interleave
        if batch_size != 2:
            # Calculate how many times to repeat each example
            repeats_per_example = batch_size // 2
            remaining = batch_size % 2

            # Apply repeat_interleave to input tensors
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    if remaining == 0:
                        # Even division
                        inputs[key] = inputs[key].repeat_interleave(
                            repeats_per_example, dim=0
                        )
                    else:
                        # Handle remainder by repeating first example one extra time
                        repeated = inputs[key].repeat_interleave(
                            repeats_per_example, dim=0
                        )
                        extra = inputs[key][:1].repeat_interleave(remaining, dim=0)
                        inputs[key] = torch.cat([repeated, extra], dim=0)

        pad_token_id = self.model.generation_config.pad_token_id
        decoder_input_ids = (
            torch.ones(
                (
                    inputs.input_ids.shape[0] * self.model.decoder.num_codebooks,
                    1,
                ),
                dtype=torch.long,
            )
            * pad_token_id
        )

        inputs["max_new_tokens"] = 1
        inputs["decoder_input_ids"] = decoder_input_ids

        return inputs
