# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
DeepFloyd IF-I-L-v1.0 model loader implementation for text-to-image generation.
"""

import torch
from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available DeepFloyd IF model variants."""

    IF_I_L_V1_0 = "IF-I-L-v1.0"


class ModelLoader(ForgeModel):
    """DeepFloyd IF-I-L-v1.0 model loader implementation for text-to-image generation."""

    _VARIANTS = {
        ModelVariant.IF_I_L_V1_0: ModelConfig(
            pretrained_model_name="DeepFloyd/IF-I-L-v1.0",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.IF_I_L_V1_0

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="DeepFloyd IF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self):
        from diffusers import DiffusionPipeline

        self._pipeline = DiffusionPipeline.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=torch.float32,
        )
        self._pipeline.to("cpu")
        return self._pipeline

    def load_model(self, *, dtype_override=None, **kwargs):
        if self._pipeline is None:
            self._load_pipeline()

        unet = self._pipeline.unet
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype_override)

        return unet

    def load_inputs(self, dtype_override=None):
        if self._pipeline is None:
            self._load_pipeline()

        dtype = dtype_override or torch.float32

        pipe = self._pipeline

        # Encode a text prompt using the pipeline's text encoder
        prompt = "a photo of a kangaroo wearing an orange hoodie and blue sunglasses"
        prompt_embeds, negative_embeds = pipe.encode_prompt(prompt)

        unet = pipe.unet
        in_channels = unet.config.in_channels
        sample_size = unet.config.sample_size

        sample = torch.randn(
            (1, in_channels, sample_size, sample_size),
            dtype=dtype,
        )

        timestep = torch.tensor([1], dtype=torch.long)

        return {
            "sample": sample.to(dtype),
            "timestep": timestep,
            "encoder_hidden_states": prompt_embeds.to(dtype),
        }
