# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Audio Open model loader implementation for text-to-audio generation
"""
import torch
from diffusers import StableAudioPipeline
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
    """Available Stable Audio model variants."""

    V1 = "v1"


class ModelLoader(ForgeModel):
    """Stable Audio Open model loader implementation for text-to-audio generation tasks."""

    _VARIANTS = {
        ModelVariant.V1: ModelConfig(
            pretrained_model_name="stabilityai/stable-audio-open-1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="StableAudioOpen",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_TTS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype_override=None):
        """Load the Stable Audio pipeline."""
        pipe_kwargs = {}
        if dtype_override is not None:
            pipe_kwargs["torch_dtype"] = dtype_override

        self.pipe = StableAudioPipeline.from_pretrained(
            self._variant_config.pretrained_model_name, **pipe_kwargs
        )

        return self.pipe

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Stable Audio transformer model."""
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        if dtype_override is not None:
            self.pipe.transformer = self.pipe.transformer.to(dtype_override)

        return self.pipe.transformer

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the Stable Audio transformer model."""
        if self.pipe is None:
            self._load_pipeline(dtype_override=dtype_override)

        dtype = dtype_override if dtype_override is not None else torch.float32
        prompt = "A gentle piano melody with soft rain in the background"
        negative_prompt = "Low quality, noise"
        num_inference_steps = 1
        audio_end_in_s = 5.0

        # Encode text prompt
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask

        prompt_embeds = self.pipe.text_encoder(
            text_input_ids, attention_mask=attention_mask
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype)

        # Encode negative prompt
        neg_text_inputs = self.pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        neg_prompt_embeds = self.pipe.text_encoder(
            neg_text_inputs.input_ids, attention_mask=neg_text_inputs.attention_mask
        )[0]
        neg_prompt_embeds = neg_prompt_embeds.to(dtype=dtype)

        # Prepare timing conditioning
        sample_rate = self.pipe.vae.config.sampling_rate
        down_ratio = self.pipe.vae.downsampling_ratio
        audio_samples = int(audio_end_in_s * sample_rate)
        latent_length = audio_samples // down_ratio

        # Create latent noise
        num_channels = self.pipe.transformer.config.in_channels
        latents = torch.randn(1, num_channels, latent_length, dtype=dtype)

        # Timestep
        timestep = torch.tensor([1.0], dtype=dtype)

        # Timing embeddings: seconds_start and seconds_total
        seconds_start = torch.tensor([0.0], dtype=dtype).unsqueeze(0)
        seconds_total = torch.tensor([audio_end_in_s], dtype=dtype).unsqueeze(0)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds,
            "global_hidden_states": neg_prompt_embeds,
            "attention_mask": attention_mask.to(dtype=dtype),
            "encoder_attention_mask": attention_mask.to(dtype=dtype),
        }
