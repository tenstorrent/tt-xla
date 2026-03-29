# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
OpenSora STDiT-v3 model loader for tt_forge_models.

STDiT3 (Spatial-Temporal Diffusion Transformer v3) is the core denoising
backbone for the Open-Sora v1.2 video generation project. It processes
latent video representations conditioned on text embeddings to generate
video from text prompts.

Reference: https://huggingface.co/hpcai-tech/OpenSora-STDiT-v3
"""

from typing import Any, Optional

import torch

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
    """Available OpenSora STDiT-v3 model variants."""

    V3 = "v3"


class ModelLoader(ForgeModel):
    """OpenSora STDiT-v3 model loader.

    Loads the STDiT3 spatial-temporal diffusion transformer for text-to-video
    generation. The model takes noisy latent video tensors, timesteps, and
    text conditioning to predict denoised outputs.
    """

    _VARIANTS = {
        ModelVariant.V3: ModelConfig(
            pretrained_model_name="hpcai-tech/OpenSora-STDiT-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="OpenSora-STDiT",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        from opensora.models.stdit.stdit3 import STDiT3

        dtype = dtype_override if dtype_override is not None else torch.float32

        model = STDiT3.from_pretrained(self._variant_config.pretrained_model_name)
        model = model.to(dtype)
        model.eval()

        return model

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

        batch_size = 1
        in_channels = 4
        num_frames = 4
        height = 32
        width = 32
        caption_channels = 4096
        seq_len = 1

        # Noisy latent video input
        x = torch.randn(
            (batch_size, in_channels, num_frames, height, width),
            dtype=dtype,
        )
        # Diffusion timestep
        timestep = torch.tensor([500.0], dtype=dtype)
        # Text conditioning embeddings
        y = torch.randn(
            (batch_size, seq_len, caption_channels),
            dtype=dtype,
        )

        return {
            "x": x,
            "timestep": timestep,
            "y": y,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if hasattr(output, "sample"):
            return output.sample
        elif isinstance(output, tuple):
            return output[0]
        return output
