# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image GGUF model loader implementation.

Loads the quantized GGUF transformer from jayn7/Z-Image-GGUF for
text-to-image generation using the Lumina2 architecture.

Available variants:
- Z_IMAGE_Q4_K_M: Q4_K_M quantized transformer (4.98 GB)
"""

from typing import Any, Optional

import torch
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

GGUF_REPO_ID = "jayn7/Z-Image-GGUF"


class ModelVariant(StrEnum):
    """Available Z-Image GGUF model variants."""

    Z_IMAGE_Q4_K_M = "Q4_K_M"


_GGUF_FILES = {
    ModelVariant.Z_IMAGE_Q4_K_M: "z_image-Q4_K_M.gguf",
}


class ModelLoader(ForgeModel):
    """Z-Image GGUF model loader.

    Loads the ZImageTransformer2DModel from a single GGUF file using
    diffusers' GGUFQuantizationConfig.
    """

    _VARIANTS = {
        ModelVariant.Z_IMAGE_Q4_K_M: ModelConfig(
            pretrained_model_name=GGUF_REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.Z_IMAGE_Q4_K_M

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

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

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load the GGUF-quantized Z-Image transformer."""
        from diffusers import GGUFQuantizationConfig, ZImageTransformer2DModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        gguf_filename = _GGUF_FILES[self._variant]
        gguf_path = hf_hub_download(
            repo_id=GGUF_REPO_ID,
            filename=gguf_filename,
        )

        self._transformer = ZImageTransformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        """Prepare synthetic transformer inputs for the Z-Image GGUF model."""
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        if self._transformer is None:
            self.load_model(dtype_override=dtype)

        batch_size = 1
        config = self._transformer.config

        latent_height = 2
        latent_width = 2

        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            1,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        encoder_hidden_states = torch.randn(
            batch_size, 8, config.caption_channels, dtype=dtype
        )

        timestep = torch.tensor([0.5], dtype=dtype).expand(batch_size)

        return {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output
