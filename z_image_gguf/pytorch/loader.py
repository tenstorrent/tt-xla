# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image GGUF model loader implementation.

Loads quantized GGUF variants of the Z-Image text-to-image
DiT transformer from unsloth/Z-Image-GGUF.
"""

from typing import Any, Optional

import torch
from diffusers import ZImagePipeline, ZImageTransformer2DModel, GGUFQuantizationConfig
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

GGUF_REPO_ID = "unsloth/Z-Image-GGUF"
PIPELINE_REPO_ID = "Tongyi-MAI/Z-Image"


class ModelVariant(StrEnum):
    """Available Z-Image GGUF model variants."""

    Z_IMAGE_GGUF_Q4_K_M = "Q4_K_M"


class ModelLoader(ForgeModel):
    """Z-Image GGUF model loader."""

    _VARIANTS = {
        ModelVariant.Z_IMAGE_GGUF_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_GGUF_Q4_K_M

    GGUF_FILES = {
        ModelVariant.Z_IMAGE_GGUF_Q4_K_M: "z-image-Q4_K_M.gguf",
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z_IMAGE_GGUF",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(self, dtype: torch.dtype = torch.bfloat16):
        """Load the GGUF-quantized transformer."""
        gguf_file = self.GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(repo_id=GGUF_REPO_ID, filename=gguf_file)
        transformer = ZImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        return transformer

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> ZImagePipeline:
        """Load the Z-Image pipeline with GGUF transformer."""
        transformer = self._load_transformer(dtype)
        self._pipe = ZImagePipeline.from_pretrained(
            PIPELINE_REPO_ID,
            transformer=transformer,
            torch_dtype=dtype,
            low_cpu_mem_usage=False,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the GGUF-quantized DiT transformer."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._pipe is None:
            self._load_pipeline(dtype)
        return self._pipe.transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare inputs for the GGUF transformer."""
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

        # The transformer expects:
        #   x: list of tensors [1, channels, 1, H, W] (one per batch)
        #   t: timestep tensor
        #   cap_feats: prompt_embeds (list of text embeddings)
        latent_input = latents.to(dtype).unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        return [latent_input_list, timestep, prompt_embeds]
