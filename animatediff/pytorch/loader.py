# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
AnimateDiff model loader for tt_forge_models.

AnimateDiff extends Stable Diffusion with temporal motion modules for
text-to-video generation. The UNetMotionModel wraps a standard UNet2D
with motion adapters that capture temporal dynamics across frames.

Reference: https://huggingface.co/camenduru/AnimateDiff
"""

from typing import Any, Optional

import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

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
    """Available AnimateDiff model variants."""

    V1_5_3 = "v1.5.3"


class ModelLoader(ForgeModel):
    """AnimateDiff UNetMotionModel loader.

    Loads the motion-augmented UNet from an AnimateDiff pipeline for
    text-to-video generation. The model combines a Stable Diffusion UNet
    with temporal motion adapter modules.
    """

    _VARIANTS = {
        ModelVariant.V1_5_3: ModelConfig(
            pretrained_model_name="guoyww/animatediff-motion-adapter-v1-5-3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.V1_5_3

    BASE_MODEL = "emilianJR/epiCRealism"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="AnimateDiff",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.float32

        adapter = MotionAdapter.from_pretrained(
            self._variant_config.pretrained_model_name,
            torch_dtype=dtype,
        )

        self.pipeline = AnimateDiffPipeline.from_pretrained(
            self.BASE_MODEL,
            motion_adapter=adapter,
            torch_dtype=dtype,
        )
        self.pipeline.scheduler = DDIMScheduler.from_pretrained(
            self.BASE_MODEL,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        return self.pipeline.unet

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.float32

        batch_size = 1
        num_frames = 16
        height = 64
        width = 64
        in_channels = 4
        cross_attention_dim = 768

        sample = torch.randn(
            (batch_size, in_channels, num_frames, height // 8, width // 8),
            dtype=dtype,
        )
        timestep = torch.randint(0, 1000, (1,))
        encoder_hidden_states = torch.randn(
            (batch_size, 77, cross_attention_dim),
            dtype=dtype,
        )

        return {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if hasattr(output, "sample"):
            return output.sample
        elif isinstance(output, tuple):
            return output[0]
        return output
