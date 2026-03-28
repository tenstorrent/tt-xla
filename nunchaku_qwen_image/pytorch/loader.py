# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Nunchaku Qwen-Image quantized text-to-image model loader.

Loads the 4-bit SVDQuant-quantized Qwen-Image transformer from
nunchaku-ai/nunchaku-qwen-image using the nunchaku inference engine.

Available variants:
- QWEN_IMAGE_INT4_R32: INT4 quantized, rank 32 (faster)
- QWEN_IMAGE_INT4_R128: INT4 quantized, rank 128 (better quality)
"""

from typing import Any, Optional

import torch
from diffusers import QwenImagePipeline
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)

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

UPSTREAM_REPO = "Qwen/Qwen-Image-2512"
NUNCHAKU_REPO = "nunchaku-ai/nunchaku-qwen-image"


class ModelVariant(StrEnum):
    """Available Nunchaku Qwen-Image model variants."""

    QWEN_IMAGE_INT4_R32 = "INT4-r32"
    QWEN_IMAGE_INT4_R128 = "INT4-r128"


class ModelLoader(ForgeModel):
    """Nunchaku Qwen-Image quantized model loader."""

    _VARIANTS = {
        ModelVariant.QWEN_IMAGE_INT4_R32: ModelConfig(
            pretrained_model_name=NUNCHAKU_REPO,
        ),
        ModelVariant.QWEN_IMAGE_INT4_R128: ModelConfig(
            pretrained_model_name=NUNCHAKU_REPO,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.QWEN_IMAGE_INT4_R32

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._pipe = None
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="NUNCHAKU_QWEN_IMAGE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_IMAGE_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_subfolder(self) -> str:
        """Return the subfolder for the selected quantization variant."""
        if self._variant == ModelVariant.QWEN_IMAGE_INT4_R128:
            return "svdq-int4-r128"
        return "svdq-int4-r32"

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> NunchakuQwenImageTransformer2DModel:
        """Load the quantized transformer from nunchaku."""
        self._transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            NUNCHAKU_REPO,
            subfolder=self._get_subfolder(),
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def _load_pipeline(self, dtype: torch.dtype = torch.bfloat16) -> QwenImagePipeline:
        """Load the full Qwen-Image pipeline with quantized transformer."""
        if self._transformer is None:
            self._load_transformer(dtype)
        self._pipe = QwenImagePipeline.from_pretrained(
            UPSTREAM_REPO,
            transformer=self._transformer,
            torch_dtype=dtype,
        )
        return self._pipe

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the quantized Qwen-Image transformer.

        Returns:
            NunchakuQwenImageTransformer2DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the transformer.

        Returns:
            dict: Input tensors matching the transformer's expected signature.
        """
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
