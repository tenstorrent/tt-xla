# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Diffusion 3.5 Large GGUF model loader implementation
"""
import torch
from typing import Optional

from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.quantizers import GGUFQuantizationConfig
from huggingface_hub import hf_hub_download

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
from ...stable_diffusion.pytorch.src.model_utils import (
    stable_diffusion_preprocessing_v35,
)


class ModelVariant(StrEnum):
    """Available Stable Diffusion 3.5 Large GGUF model variants."""

    SD3_5_LARGE_Q8_0 = "Q8_0"


class ModelLoader(ForgeModel):
    """Stable Diffusion 3.5 Large GGUF model loader implementation."""

    _VARIANTS = {
        ModelVariant.SD3_5_LARGE_Q8_0: ModelConfig(
            pretrained_model_name="city96/stable-diffusion-3.5-large-gguf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SD3_5_LARGE_Q8_0

    _GGUF_FILES = {
        ModelVariant.SD3_5_LARGE_Q8_0: "sd3.5_large-Q8_0.gguf",
    }

    BASE_PIPELINE = "stabilityai/stable-diffusion-3.5-large"

    prompt = "An astronaut riding a green horse"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.gguf_file = self._GGUF_FILES[self._variant]
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Stable Diffusion 3.5 Large GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the SD3.5 Large transformer from a GGUF quantized file.

        Returns:
            torch.nn.Module: The SD3 transformer instance.
        """
        gguf_path = hf_hub_download(
            self._variant_config.pretrained_model_name,
            filename=self.gguf_file,
        )

        compute_dtype = dtype_override or torch.bfloat16

        transformer = SD3Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
            torch_dtype=compute_dtype,
        )

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.BASE_PIPELINE,
            transformer=transformer,
            torch_dtype=compute_dtype,
        )
        self.pipeline.to("cpu")

        return transformer

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the SD3.5 Large GGUF model.

        Returns:
            list: [latent_model_input, timestep, prompt_embeds, pooled_prompt_embeds]
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
