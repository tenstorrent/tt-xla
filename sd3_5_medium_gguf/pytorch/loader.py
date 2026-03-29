# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SD3.5 Medium GGUF (calcuis/sd3.5-medium-gguf) model loader implementation.

Stable Diffusion 3.5 Medium is a text-to-image generation model in GGUF quantized format,
based on the SD3 MMDiT transformer architecture with 2B parameters.

Available variants:
- SD3_5_MEDIUM_Q4_K_M: Q4_K_M quantized variant
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
from .src.model_utils import load_gguf_pipe, stable_diffusion_preprocessing_v35

REPO_ID = "calcuis/sd3.5-medium-gguf"


class ModelVariant(StrEnum):
    """Available SD3.5 Medium GGUF model variants."""

    SD3_5_MEDIUM_Q4_K_M = "sd3.5_medium_Q4_K_M"


class ModelLoader(ForgeModel):
    """SD3.5 Medium GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.SD3_5_MEDIUM_Q4_K_M: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SD3_5_MEDIUM_Q4_K_M

    GGUF_FILE = "sd3.5-medium-Q4_K_M.gguf"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SD3.5 Medium GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SD3.5 Medium pipeline from GGUF checkpoint.

        Returns:
            DiffusionPipeline: The loaded pipeline instance.
        """
        if self.pipeline is None:
            self.pipeline = load_gguf_pipe(REPO_ID, self.GGUF_FILE)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model.

        Returns:
            list: Input tensors for the transformer model.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        (
            latent_model_input,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
        ) = stable_diffusion_preprocessing_v35(self.pipeline, self.prompt)

        if dtype_override:
            latent_model_input = latent_model_input.to(dtype_override)
            timestep = timestep.to(dtype_override)
            prompt_embeds = prompt_embeds.to(dtype_override)
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype_override)

        return [latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds]
