# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
LatentSync-1.5 UNet model loader implementation.

LatentSync is an audio-conditioned latent diffusion model for lip synchronization.
It generates lip-synced video by conditioning a temporal UNet on audio embeddings
from Whisper. The UNet is based on Stable Diffusion 1.5 with AnimateDiff-style
temporal attention modules and additional audio cross-attention layers.

Repository: https://huggingface.co/ByteDance/LatentSync-1.5
"""

from typing import Any, Optional

import torch
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


class ModelVariant(StrEnum):
    """Available LatentSync model variants."""

    LATENTSYNC_1_5 = "1.5"


class ModelLoader(ForgeModel):
    """
    LatentSync UNet model loader.

    Loads the temporal UNet component of LatentSync-1.5, which processes
    noised video latents concatenated with reference latents and a binary mask,
    conditioned on audio embeddings via cross-attention.
    """

    _VARIANTS = {
        ModelVariant.LATENTSYNC_1_5: ModelConfig(
            pretrained_model_name="ByteDance/LatentSync-1.5",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LATENTSYNC_1_5

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="LatentSync",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load the LatentSync UNet model.

        Downloads the UNet checkpoint from HuggingFace Hub and loads it into
        a UNet3DConditionModel (AnimateDiff-style temporal UNet based on SD 1.5).

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            torch.nn.Module: The LatentSync UNet model instance.
        """
        from diffusers import UNet3DConditionModel

        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        repo_id = self._variant_config.pretrained_model_name

        # LatentSync UNet extends SD 1.5 UNet with AnimateDiff temporal modules
        # and audio cross-attention. Config mirrors SD 1.5 with 3D block types
        # and modified in_channels for reference+mask concatenation.
        # Input: noised latent (4ch) + reference latent (4ch) + mask (1ch) = 9ch
        unet = UNet3DConditionModel(
            sample_size=64,
            in_channels=9,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            up_block_types=(
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ),
            block_out_channels=(320, 640, 1280, 1280),
            layers_per_block=2,
            cross_attention_dim=768,
        )

        # Download and load the LatentSync UNet checkpoint
        ckpt_path = hf_hub_download(repo_id=repo_id, filename="latentsync_unet.pt")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # strict=False: LatentSync adds custom audio cross-attention layers
        # that may not exist in the standard UNet3DConditionModel
        unet.load_state_dict(state_dict, strict=False)
        unet.eval()

        if dtype_override is not None:
            unet = unet.to(dtype=dtype_override)

        self._unet_config = unet.config
        return unet

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load synthetic inputs for the LatentSync UNet.

        The UNet expects:
        - sample: noised latent (4ch) + reference latent (4ch) + mask (1ch) = 9ch
          with temporal dimension for video frames
        - timestep: diffusion timestep
        - encoder_hidden_states: audio embeddings from Whisper (projected to 768-dim)

        Args:
            dtype_override: Optional torch.dtype to override the input dtype.
            batch_size: Batch size for the inputs.

        Returns:
            dict: Dictionary of input tensors for the UNet forward pass.
        """
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        num_frames = 2
        height = 8
        width = 8

        # Video latent: (batch, channels, frames, height, width)
        sample = torch.randn(batch_size, 9, num_frames, height, width, dtype=dtype)

        # Audio encoder hidden states projected to cross_attention_dim
        encoder_hidden_states = torch.randn(batch_size, 16, 768, dtype=dtype)

        timestep = torch.tensor([0], dtype=torch.long).expand(batch_size)

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "return_dict": False,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
