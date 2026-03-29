# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ByteDance SeedVR2-7B video restoration model loader implementation.

SeedVR2 is a one-step diffusion-based video restoration model using a NaDiT
(Neighborhood Attention Diffusion Transformer) architecture. It takes degraded
low-quality video frames and produces high-quality restored output.

Available variants:
- SEEDVR2_7B: Standard 7B parameter model (seedvr2_ema_7b.pth)
- SEEDVR2_7B_SHARP: Sharper variant (seedvr2_ema_7b_sharp.pth)
"""

import sys
from typing import Optional

import torch
from huggingface_hub import snapshot_download

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

REPO_ID = "ByteDance-Seed/SeedVR2-7B"

# NaDiT model input dimensions for testing
# The model operates in latent space: patch_size [1, 2, 2],
# VAE compression 4x temporal / 8x spatial, 16 latent channels
LATENT_CHANNELS = 16
CONDITION_CHANNELS = 16
MASK_CHANNELS = 1
INPUT_CHANNELS = LATENT_CHANNELS + CONDITION_CHANNELS + MASK_CHANNELS  # 33
LATENT_HEIGHT = 8
LATENT_WIDTH = 8
LATENT_DEPTH = 4  # temporal frames in latent space


class ModelVariant(StrEnum):
    """Available SeedVR2 model variants."""

    SEEDVR2_7B = "7B"
    SEEDVR2_7B_SHARP = "7B_Sharp"


class ModelLoader(ForgeModel):
    """ByteDance SeedVR2-7B video restoration model loader."""

    _VARIANTS = {
        ModelVariant.SEEDVR2_7B: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
        ModelVariant.SEEDVR2_7B_SHARP: ModelConfig(
            pretrained_model_name=REPO_ID,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SEEDVR2_7B

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._repo_path = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SeedVR2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _get_repo_path(self):
        """Download the SeedVR2 repository and return its local path."""
        if self._repo_path is None:
            self._repo_path = snapshot_download(repo_id=REPO_ID)
        return self._repo_path

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the SeedVR2-7B NaDiT model.

        Downloads the model repository, imports the custom NaDiT architecture,
        and loads the pretrained weights.

        Returns:
            torch.nn.Module: The NaDiT diffusion transformer model.
        """
        from omegaconf import OmegaConf

        repo_path = self._get_repo_path()

        # Add repo to path for custom module imports
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        from utils.utils import instantiate_from_config

        config = OmegaConf.load(f"{repo_path}/configs_7b/main.yaml")

        # Instantiate the NaDiT model from config
        model = instantiate_from_config(config.model.dit)

        # Load pretrained weights
        if self._variant == ModelVariant.SEEDVR2_7B_SHARP:
            ckpt_path = f"{repo_path}/seedvr2_ema_7b_sharp.pth"
        else:
            ckpt_path = f"{repo_path}/seedvr2_ema_7b.pth"

        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

        if dtype_override is not None:
            model = model.to(dtype=dtype_override)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load synthetic latent-space inputs for the SeedVR2 NaDiT model.

        Creates synthetic inputs matching the model's expected format:
        - x: noised latent tensor [batch, channels, depth, height, width]
        - timestep: diffusion timestep
        - context: text conditioning embeddings
        - y: pooled text embeddings

        Returns:
            dict: Input tensors for the NaDiT forward pass.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        repo_path = self._get_repo_path()

        # Load pre-computed text embeddings
        pos_emb = torch.load(
            f"{repo_path}/pos_emb.pt", map_location="cpu", weights_only=True
        )

        # x: concatenated [noise, condition_latents, mask] along channel dim
        x = torch.randn(
            1, INPUT_CHANNELS, LATENT_DEPTH, LATENT_HEIGHT, LATENT_WIDTH, dtype=dtype
        )

        # Single-step diffusion: timestep = 1.0
        timestep = torch.tensor([1.0], dtype=dtype)

        # Text context from pre-computed embeddings (dim 5120)
        if isinstance(pos_emb, dict):
            context = pos_emb.get("context", torch.randn(1, 77, 5120, dtype=dtype))
            y = pos_emb.get("y", torch.randn(1, 5120, dtype=dtype))
        else:
            context = pos_emb if pos_emb.dim() == 3 else pos_emb.unsqueeze(0)
            y = torch.randn(1, 5120, dtype=dtype)

        if dtype_override is not None:
            context = context.to(dtype=dtype_override)
            y = y.to(dtype=dtype_override)

        return {"x": x, "timestep": timestep, "context": context, "y": y}
