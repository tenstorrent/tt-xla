# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Diffusion Forcing Transformer (DFoT) model loader implementation.

DFoT is a video diffusion model that generates videos conditioned on context frames.
It uses per-token noise level conditioning, enabling autoregressive-style generation
within a diffusion framework.

Reference: https://github.com/kwsong0113/diffusion-forcing-transformer
HuggingFace: https://huggingface.co/kiwhansong/DFoT
"""

from typing import Optional

import torch
from huggingface_hub import hf_hub_download  # type: ignore[import]

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
from .src.model import DiT3D, DiT3DConfig, load_dit3d_from_checkpoint

# HuggingFace repo hosting the pretrained checkpoints.
HF_REPO_ID = "kiwhansong/DFoT"


class ModelVariant(StrEnum):
    """Available DFoT model variants."""

    DFOT_RE10K = "RE10K"
    DFOT_K600 = "K600"
    DFOT_MCRAFT = "MCRAFT"


class ModelLoader(ForgeModel):
    """DFoT (Diffusion Forcing Transformer) model loader.

    Loads the DiT3D backbone from pretrained PyTorch Lightning checkpoints
    hosted on HuggingFace. Each variant corresponds to a different training
    dataset:
      - RE10K: RealEstate10K (camera-pose conditioned video generation)
      - K600: Kinetics-600 (unconditional video generation)
      - MCRAFT: Minecraft (unconditional video generation)
    """

    _VARIANTS = {
        ModelVariant.DFOT_RE10K: ModelConfig(
            pretrained_model_name="pretrained_models/DFoT_RE10K.ckpt",
        ),
        ModelVariant.DFOT_K600: ModelConfig(
            pretrained_model_name="pretrained_models/DFoT_K600.ckpt",
        ),
        ModelVariant.DFOT_MCRAFT: ModelConfig(
            pretrained_model_name="pretrained_models/DFoT_MCRAFT.ckpt",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DFOT_RE10K

    # Default DiT-XL configuration used by all DFoT pretrained models.
    _DEFAULT_CFG = DiT3DConfig(
        in_channels=4,
        patch_size=2,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        max_tokens=16,
        spatial_resolution=32,
        external_cond_dim=0,
    )

    # Number of video frames for sample inputs.
    DEFAULT_NUM_FRAMES = 8

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="DFoT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the DiT3D backbone from a pretrained checkpoint.

        Args:
            dtype_override: Optional torch.dtype to convert the model to.

        Returns:
            DiT3D: The loaded DiT3D backbone in eval mode.
        """
        ckpt_filename = self._variant_config.pretrained_model_name

        # Download checkpoint from HuggingFace
        ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=ckpt_filename)

        # Load model from checkpoint
        model = load_dit3d_from_checkpoint(ckpt_path, self._DEFAULT_CFG)
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, num_frames=None):
        """Load sample inputs for the DiT3D backbone.

        Generates random noisy latent frames and noise levels suitable for
        a single denoising step.

        Args:
            dtype_override: Optional torch.dtype for the input tensors.
            num_frames: Number of video frames (default: DEFAULT_NUM_FRAMES).

        Returns:
            list: [x, noise_levels] where:
                - x: (1, T, 4, 32, 32) noisy latent video frames
                - noise_levels: (1, T) per-frame noise levels
        """
        cfg = self._DEFAULT_CFG
        T = num_frames or self.DEFAULT_NUM_FRAMES
        dtype = dtype_override or torch.float32

        # Random noisy latent frames
        x = torch.randn(
            1,
            T,
            cfg.in_channels,
            cfg.spatial_resolution,
            cfg.spatial_resolution,
            dtype=dtype,
        )

        # Random noise levels (logSNR values, typically in [-20, 20] range)
        noise_levels = torch.randn(1, T, dtype=dtype)

        return [x, noise_levels]
