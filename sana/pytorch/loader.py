# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sana model loader implementation for text-to-image generation
"""
import torch
from diffusers import SanaPipeline
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
    """Available Sana model variants."""

    TINY_RANDOM = "Tiny_Random"


class ModelLoader(ForgeModel):
    """Sana model loader implementation for text-to-image generation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.TINY_RANDOM: ModelConfig(
            pretrained_model_name="optimum-intel-internal-testing/tiny-random-sana",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.TINY_RANDOM

    # Shared configuration parameters
    prompt = "A cat sitting on a windowsill"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="Sana",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load the Sana pipeline for the current variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            The loaded pipeline instance
        """
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = SanaPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Sana transformer model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The Sana transformer model instance for text-to-image generation.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Sana transformer model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            dict: Input tensors that can be fed to the transformer model.
        """
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32

        # Get model config
        transformer = self.pipe.transformer
        in_channels = transformer.config.in_channels
        sample_size = transformer.config.sample_size

        # Create latent input
        hidden_states = torch.randn(
            batch_size, in_channels, sample_size, sample_size, dtype=dtype
        )

        # Create timestep
        timestep = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        # Encode prompt to get text embeddings
        text_inputs = self.pipe.tokenizer(
            self.prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_output = self.pipe.text_encoder(text_inputs.input_ids)
            encoder_hidden_states = encoder_output.last_hidden_state.to(dtype=dtype)

        encoder_hidden_states = encoder_hidden_states.expand(batch_size, -1, -1)
        encoder_attention_mask = text_inputs.attention_mask.expand(batch_size, -1)

        inputs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
        }

        return inputs
