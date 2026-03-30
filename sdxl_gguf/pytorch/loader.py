# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SDXL GGUF (hum-ma/SDXL-models-GGUF) model loader implementation.

Stable Diffusion XL is a text-to-image generation model in GGUF quantized format,
based on the SDXL UNet architecture with 3B parameters.

Available variants:
- SDXL_1_0_Q4_0: Q4_0 quantized SDXL 1.0 base model
"""

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
from .src.model_utils import load_sdxl_gguf_pipe, stable_diffusion_preprocessing_xl

REPO_ID = "hum-ma/SDXL-models-GGUF"


class ModelVariant(StrEnum):
    """Available SDXL GGUF model variants."""

    SDXL_1_0_Q4_0 = "sdxl_1.0_Q4_0"


class ModelLoader(ForgeModel):
    """SDXL GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.SDXL_1_0_Q4_0: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SDXL_1_0_Q4_0

    GGUF_FILE = "sdxl-1.0-base-Q4_0.gguf"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SDXL GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SDXL pipeline from GGUF checkpoint.

        Returns:
            DiffusionPipeline: The loaded pipeline instance.
        """
        if self.pipeline is None:
            self.pipeline = load_sdxl_gguf_pipe(REPO_ID, self.GGUF_FILE)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Returns:
            list: Input tensors for the UNet model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timesteps,
            prompt_embeds,
            timestep_cond,
            added_cond_kwargs,
            add_time_ids,
        ) = stable_diffusion_preprocessing_xl(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timesteps = timesteps.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)

        return [latent_model_input, timesteps, prompt_embeds, added_cond_kwargs]
