# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Wan Fun Control model loader for tt_forge_models.

Wan2.1-Fun-14B-Control is a controllable video generation model from alibaba-pai
that supports control conditions such as Canny, Depth, Pose, and MLSD.
It uses a WanTransformer3DModel with 36 input channels (16 latent + 20 control/mask).

Repository: https://huggingface.co/alibaba-pai/Wan2.1-Fun-14B-Control
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
from .src.utils import (
    load_transformer,
    load_transformer_inputs,
)


class ModelVariant(StrEnum):
    """Available Wan Fun Control variants."""

    WAN21_FUN_14B_CONTROL = "2.1_Fun_14B_Control"


class ModelLoader(ForgeModel):
    """
    Loader for alibaba-pai/Wan2.1-Fun-14B-Control video generation model.

    Loads the WanTransformer3DModel with 36 input channels for control
    conditioning. The model stores weights at the repo root (no subfolder).
    """

    _VARIANTS = {
        ModelVariant.WAN21_FUN_14B_CONTROL: ModelConfig(
            pretrained_model_name="alibaba-pai/Wan2.1-Fun-14B-Control",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.WAN21_FUN_14B_CONTROL

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
    ):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="WanFunControl",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.MM_VIDEO_TTT,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return load_transformer(self._variant_config.pretrained_model_name, dtype)

    def load_inputs(self, dtype_override=None, **kwargs) -> Any:
        dtype = dtype_override if dtype_override is not None else torch.bfloat16
        return load_transformer_inputs(dtype)

    def unpack_forward_output(self, output: Any) -> torch.Tensor:
        if isinstance(output, tuple):
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        return output
