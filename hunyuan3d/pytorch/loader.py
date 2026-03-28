# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D-2.1 model loader implementation for image-to-3D generation.

Loads the HunYuanDiTPlain denoiser (DiT backbone) from the Hunyuan3D pipeline,
which is the core flow-matching transformer for generating 3D shapes from
image conditioning via a DINOv2-large encoder.

Requires the Hunyuan3D-2.1 repository to be cloned at /tmp/hunyuan3d_repo.
"""
import os
import sys

import torch
from huggingface_hub import hf_hub_download
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

HUNYUAN3D_REPO_PATH = "/tmp/hunyuan3d_repo"


def _ensure_hunyuan3d_importable():
    """Ensure the Hunyuan3D-2.1 repo is cloned and importable."""
    if HUNYUAN3D_REPO_PATH not in sys.path:
        if not os.path.isdir(HUNYUAN3D_REPO_PATH):
            import subprocess

            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git",
                    HUNYUAN3D_REPO_PATH,
                ]
            )

        sys.path.insert(0, HUNYUAN3D_REPO_PATH)


class ModelVariant(StrEnum):
    """Available Hunyuan3D model variants."""

    DIT_FLOW_MATCHING = "DiT_Flow_Matching"


class ModelLoader(ForgeModel):
    """Hunyuan3D model loader for the HunYuanDiTPlain denoiser (DiT backbone)."""

    _VARIANTS = {
        ModelVariant.DIT_FLOW_MATCHING: ModelConfig(
            pretrained_model_name="andreca/hunyuan3d-2.1xet",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.DIT_FLOW_MATCHING

    # DiT model config from the model card
    _HIDDEN_SIZE = 2048
    _DEPTH = 21
    _NUM_HEADS = 16
    _NUM_LATENTS = 4096
    _IN_CHANNELS = 64
    _COND_CHANNELS = 1024
    _COND_SEQ_LEN = 257  # DINOv2 ViT-L/14: 1 CLS + 256 patch tokens

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="Hunyuan3D",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the Hunyuan3D DiT denoiser model.

        Returns:
            torch.nn.Module: The HunYuanDiTPlain flow-matching denoiser.
        """
        _ensure_hunyuan3d_importable()
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        repo_id = self._variant_config.pretrained_model_name

        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(repo_id)
        model = pipeline.model
        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load sample inputs for the HunYuanDiTPlain denoiser.

        Returns:
            dict: Input tensors (x, t, cond) for the model forward pass.
        """
        dtype = dtype_override or torch.float32

        # x: noisy latent [B, num_latents, in_channels]
        x = torch.randn(
            batch_size,
            self._NUM_LATENTS,
            self._IN_CHANNELS,
            dtype=dtype,
        )

        # t: diffusion timestep
        t = torch.full((batch_size,), 0.5, dtype=dtype)

        # cond: DINOv2 image conditioning tokens [B, seq_len, cond_channels]
        cond = torch.randn(
            batch_size,
            self._COND_SEQ_LEN,
            self._COND_CHANNELS,
            dtype=dtype,
        )

        return {"x": x, "t": t, "cond": cond}
