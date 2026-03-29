# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan2.1-Fun-1.3B-Control model loader implementation.

Loads the diffusion transformer from alibaba-pai/Wan2.1-Fun-1.3B-Control.
This is a controllable video generation model from the VideoX-Fun project,
based on the Wan 2.1 1.3B architecture with control signal conditioning
(Canny edges, depth maps, pose, etc.).

The transformer weights are stored as a single safetensors file. We load
them using diffusers' WanTransformer3DModel with a config reference to the
standard Wan 2.1 1.3B architecture.

Repository: https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-Control

Available variants:
- WAN21_FUN_1_3B_CONTROL: Wan 2.1 Fun 1.3B Control transformer
"""

from typing import Any, Optional

import torch
from diffusers import WanTransformer3DModel
from huggingface_hub import hf_hub_download

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

REPO_ID = "alibaba-pai/Wan2.1-Fun-1.3B-Control"
SAFETENSORS_FILE = "diffusion_pytorch_model.safetensors"

# Reference config repo for the Wan 1.3B transformer architecture
WAN_CONFIG_REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"


class ModelVariant(StrEnum):
    """Available Wan2.1-Fun-1.3B-Control model variants."""

    WAN21_FUN_1_3B_CONTROL = "1.3B-Control"


class ModelLoader(ForgeModel):
    """Wan2.1-Fun-1.3B-Control model loader for the diffusion transformer."""

    _VARIANTS = {
        ModelVariant.WAN21_FUN_1_3B_CONTROL: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }
    DEFAULT_VARIANT = ModelVariant.WAN21_FUN_1_3B_CONTROL

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._transformer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="WAN_FUN_CONTROL",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_transformer(
        self, dtype: torch.dtype = torch.bfloat16
    ) -> WanTransformer3DModel:
        """Load the diffusion transformer from the single-file checkpoint."""
        ckpt_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=SAFETENSORS_FILE,
        )

        self._transformer = WanTransformer3DModel.from_single_file(
            ckpt_path,
            config=WAN_CONFIG_REPO,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        self._transformer.eval()
        return self._transformer

    def load_model(self, *, dtype_override: Optional[torch.dtype] = None, **kwargs):
        """Load and return the Wan2.1-Fun-1.3B-Control diffusion transformer.

        Returns:
            WanTransformer3DModel instance.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        if self._transformer is None:
            return self._load_transformer(dtype)
        if dtype_override is not None:
            self._transformer = self._transformer.to(dtype=dtype_override)
        return self._transformer

    def load_inputs(self, **kwargs) -> Any:
        """Prepare sample inputs for the diffusion transformer.

        Returns:
            dict: Input tensors matching WanTransformer3DModel forward signature.
        """
        dtype = kwargs.get("dtype_override", torch.bfloat16)
        batch_size = 1

        if self._transformer is None:
            self.load_model(dtype_override=dtype)

        config = self._transformer.config

        # Small latent dimensions for testing
        # Wan VAE compression: 4x temporal, 8x spatial
        latent_num_frames = 2
        latent_height = 4
        latent_width = 4

        # Video hidden states: [B, C, T, H, W]
        hidden_states = torch.randn(
            batch_size,
            config.in_channels,
            latent_num_frames,
            latent_height,
            latent_width,
            dtype=dtype,
        )

        # Text encoder hidden states: [B, seq_len, text_dim]
        text_seq_len = 64
        encoder_hidden_states = torch.randn(
            batch_size,
            text_seq_len,
            config.text_dim,
            dtype=dtype,
        )

        # Diffusion timestep
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
        if hasattr(output, "sample"):
            return output.sample
        return output
