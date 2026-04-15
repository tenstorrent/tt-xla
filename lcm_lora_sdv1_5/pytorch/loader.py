# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LCM-LoRA SDv1.5 model loader implementation.

Loads a Stable Diffusion v1.5 base pipeline and applies LCM-LoRA weights
from latent-consistency/lcm-lora-sdv1-5 for fast 2-8 step text-to-image
generation using Latent Consistency Models.
"""

from typing import Any, Optional

import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler  # type: ignore[import]

from ...base import ForgeModel
from ...config import (
    Framework,
    ModelConfig,
    ModelGroup,
    ModelInfo,
    ModelSource,
    ModelTask,
    StrEnum,
)

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_REPO = "latent-consistency/lcm-lora-sdv1-5"


class ModelVariant(StrEnum):
    """Available LCM-LoRA SDv1.5 variants."""

    LCM_LORA_SDV1_5 = "LCM_LoRA_SDv1.5"


class ModelLoader(ForgeModel):
    """LCM-LoRA SDv1.5 model loader."""

    _VARIANTS = {
        ModelVariant.LCM_LORA_SDV1_5: ModelConfig(
            pretrained_model_name=BASE_MODEL,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.LCM_LORA_SDV1_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline: Optional[AutoPipelineForText2Image] = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="LCM_LORA_SDV1_5",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the SD v1.5 pipeline with LCM-LoRA weights and return the UNet."""
        dtype = dtype_override if dtype_override is not None else torch.float32

        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        # Swap scheduler for LCM compatibility
        self.pipeline.scheduler = LCMScheduler.from_config(
            self.pipeline.scheduler.config
        )

        # Load and fuse LCM-LoRA weights
        self.pipeline.load_lora_weights(LORA_REPO)
        self.pipeline.fuse_lora()

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs):
        """Prepare preprocessed tensor inputs for the UNet."""
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        dtype = self.pipeline.unet.dtype

        prompt = (
            "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k"
        )

        # Encode text prompt into embeddings
        text_inputs = self.pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            encoder_hidden_states = self.pipeline.text_encoder(
                text_inputs.input_ids
            )[0].to(dtype)

        # Create latent noise input and timestep
        in_channels = self.pipeline.unet.config.in_channels
        sample_size = self.pipeline.unet.config.sample_size
        latent_sample = torch.randn(
            1, in_channels, sample_size, sample_size, dtype=dtype
        )
        timestep = torch.tensor([1.0], dtype=dtype)

        if dtype_override:
            latent_sample = latent_sample.to(dtype_override)
            timestep = timestep.to(dtype_override)
            encoder_hidden_states = encoder_hidden_states.to(dtype_override)

        return [latent_sample, timestep, encoder_hidden_states]

    def unpack_forward_output(self, fwd_output: Any) -> torch.Tensor:
        """Unpack UNet output tuple to the sample tensor."""
        if isinstance(fwd_output, tuple):
            return fwd_output[0]
        if hasattr(fwd_output, "sample"):
            return fwd_output.sample
        return fwd_output
