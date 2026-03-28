# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Stable Virtual Camera model loader for tt_forge_models.

Stable Virtual Camera (SeVa) is a 1.3B parameter diffusion model for novel view
synthesis. It generates 3D-consistent novel views of a scene from input images
and specified camera trajectories at 576p resolution.

Repository: https://huggingface.co/stabilityai/stable-virtual-camera
"""

from typing import Any, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

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
    """Available Stable Virtual Camera variants."""

    V1_0 = "v1.0"
    V1_1 = "v1.1"


class ModelLoader(ForgeModel):
    """
    Loader for Stable Virtual Camera (SeVa) novel view synthesis model.

    The model is a U-Net-style diffusion model with multi-view transformer
    attention that generates novel camera viewpoints from input images.
    """

    _VARIANTS = {
        ModelVariant.V1_0: ModelConfig(
            pretrained_model_name="stabilityai/stable-virtual-camera",
        ),
        ModelVariant.V1_1: ModelConfig(
            pretrained_model_name="stabilityai/stable-virtual-camera",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_1

    _WEIGHT_NAMES = {
        ModelVariant.V1_0: "model.safetensors",
        ModelVariant.V1_1: "modelv1.1.safetensors",
    }

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="StableVirtualCamera",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from seva.model import Seva, SevaParams
        from seva.model import SGMWrapper

        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        weight_name = self._WEIGHT_NAMES[self._variant]
        weight_path = hf_hub_download(
            self._variant_config.pretrained_model_name,
            filename=weight_name,
        )
        state_dict = load_file(weight_path)

        model = Seva(SevaParams()).to(dtype)
        model.load_state_dict(state_dict)
        model = SGMWrapper(model.eval())

        return model

    def load_inputs(self, *, dtype_override=None, **kwargs) -> dict:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16

        batch_size = 1
        num_frames = 21
        model_channels = 320
        in_channels = 11
        height = 72  # 576 / 8 (VAE downscale factor)
        width = 72

        # Noisy latent input
        x = torch.randn(batch_size, num_frames, in_channels, height, width, dtype=dtype)

        # Diffusion timestep sigma
        sigma = torch.tensor([1.0], dtype=dtype).expand(batch_size)

        return {
            "x": x,
            "sigma": sigma,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        return output
