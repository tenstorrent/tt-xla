# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo FP8 model loader implementation.

Loads the FP8-quantized Z-Image-Turbo diffusion transformer from
drbaph/Z-Image-Turbo-FP8.

Available variants:
- E4M3FN: FP8 E4M3FN quantized transformer
- E5M2: FP8 E5M2 quantized transformer
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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

REPO_ID = "drbaph/Z-Image-Turbo-FP8"
BASE_REPO_ID = "Tongyi-MAI/Z-Image-Turbo"


class ModelVariant(StrEnum):
    """Available Z-Image-Turbo FP8 model variants."""

    E4M3FN = "E4M3FN"
    E5M2 = "E5M2"


VARIANT_FILENAMES = {
    ModelVariant.E4M3FN: "z_image_turbo_fp8_e4m3fn.safetensors",
    ModelVariant.E5M2: "z_image_turbo_fp8_e5m2.safetensors",
}


class ModelLoader(ForgeModel):
    """Z-Image-Turbo FP8 model loader."""

    _VARIANTS = {
        ModelVariant.E4M3FN: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.E5M2: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.E4M3FN

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_TURBO_FP8",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the base Z-Image-Turbo pipeline and swap in FP8 weights."""
        # Load base pipeline
        self._pipe = ZImagePipeline.from_pretrained(
            BASE_REPO_ID,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )

        # Download and load the FP8 quantized transformer weights
        filename = VARIANT_FILENAMES[self._variant]
        fp8_path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        state_dict = load_file(fp8_path)
        self._pipe.transformer.load_state_dict(state_dict)
        self._pipe.transformer.eval()
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the FP8-quantized transformer.

        Returns the DiT transformer component with FP8 weights loaded.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        if dtype_override is not None:
            self._pipe.transformer = self._pipe.transformer.to(dtype_override)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the FP8 transformer variant."""
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        height = 128
        width = 128
        prompt = "A photo of an astronaut riding a horse on mars"

        if self._pipe is None:
            self._load_pipeline(dtype)

        # Encode the prompt
        prompt_embeds, _ = self._pipe.encode_prompt(
            prompt=prompt,
            device="cpu",
            do_classifier_free_guidance=False,
        )

        # Prepare latents
        num_channels_latents = self._pipe.transformer.in_channels
        vae_scale = self._pipe.vae_scale_factor * 2
        latent_h = height // vae_scale
        latent_w = width // vae_scale
        latents = torch.randn(
            1, num_channels_latents, latent_h, latent_w, dtype=torch.float32
        )

        # Prepare timestep
        timestep = torch.tensor([0.5], dtype=dtype)

        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
