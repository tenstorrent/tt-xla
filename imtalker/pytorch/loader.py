# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
IMTalker renderer model loader implementation for talking face generation tasks.
"""
import sys
import types

import torch
import torch.nn as nn
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


class IMTRendererWrapper(nn.Module):
    """Wrapper around the IMTalker renderer for video-driven face reenactment.

    Exposes a clean forward pass that takes a current driving frame and a
    reference frame, producing the reenacted output frame.
    """

    def __init__(self, renderer):
        super().__init__()
        self.renderer = renderer

    def forward(self, x_current, x_reference):
        output_frame, _ = self.renderer(x_current, x_reference)
        return output_frame


class ModelVariant(StrEnum):
    """Available IMTalker model variants."""

    RENDERER = "renderer"


class ModelLoader(ForgeModel):
    """IMTalker renderer model loader for talking face generation."""

    _VARIANTS = {
        ModelVariant.RENDERER: ModelConfig(
            pretrained_model_name="cbsjtu01/IMTalker",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.RENDERER

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="IMTalker",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the IMTalker renderer model."""
        from huggingface_hub import hf_hub_download, snapshot_download

        # Download the Space source code containing the renderer/generator modules
        space_path = snapshot_download(repo_id="chenxie95/IMTalker", repo_type="space")
        if space_path not in sys.path:
            sys.path.insert(0, space_path)

        from renderer.models import IMTRenderer

        # Build a minimal config object with the attributes the renderer needs
        args = types.SimpleNamespace(
            swin_res_threshold=128,
            window_size=8,
            num_heads=8,
        )

        renderer = IMTRenderer(args)

        # Download and load the renderer checkpoint
        ckpt_path = hf_hub_download(
            repo_id="cbsjtu01/IMTalker", filename="renderer.ckpt"
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {
            k.replace("gen.", ""): v
            for k, v in state_dict.items()
            if k.startswith("gen.")
        }
        renderer.load_state_dict(clean_state_dict, strict=False)

        if dtype_override is not None:
            renderer = renderer.to(dtype=dtype_override)

        renderer.eval()

        model = IMTRendererWrapper(renderer)
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Load sample inputs for the IMTalker renderer.

        Returns a pair of 512x512 image tensors (current driving frame,
        reference frame).
        """
        dtype = dtype_override or torch.float32
        x_current = torch.randn(1, 3, 512, 512, dtype=dtype)
        x_reference = torch.randn(1, 3, 512, 512, dtype=dtype)
        return (x_current, x_reference)
