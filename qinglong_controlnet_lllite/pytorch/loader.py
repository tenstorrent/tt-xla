# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ControlNet-LLLite model loader implementation

ControlNet-LLLite is a lightweight ControlNet variant for Stable Diffusion XL
by bdsqlsz. It provides conditional control (canny, depth, openpose, etc.)
using small modules that patch into the SDXL UNet.
"""

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
from .src.model_utils import load_controlnet_lllite, create_dummy_input


class ModelVariant(StrEnum):
    """Available ControlNet-LLLite model variants."""

    CANNY = "Canny"
    DEPTH = "Depth"
    DEPTH_V2 = "Depth_V2"
    DW_OPENPOSE = "DW_OpenPose"
    SOFTEDGE = "Softedge"
    TILE_REALISTIC = "Tile_Realistic"


REPO_ID = "bdsqlsz/qinglong_controlnet-lllite"


class ModelLoader(ForgeModel):
    """ControlNet-LLLite model loader implementation."""

    _VARIANTS = {
        ModelVariant.CANNY: ModelConfig(
            pretrained_model_name="bdsqlsz_controlllite_xl_canny.safetensors",
        ),
        ModelVariant.DEPTH: ModelConfig(
            pretrained_model_name="bdsqlsz_controlllite_xl_depth.safetensors",
        ),
        ModelVariant.DEPTH_V2: ModelConfig(
            pretrained_model_name="bdsqlsz_controlllite_xl_depth_V2.safetensors",
        ),
        ModelVariant.DW_OPENPOSE: ModelConfig(
            pretrained_model_name="bdsqlsz_controlllite_xl_dw_openpose.safetensors",
        ),
        ModelVariant.SOFTEDGE: ModelConfig(
            pretrained_model_name="bdsqlsz_controlllite_xl_softedge.safetensors",
        ),
        ModelVariant.TILE_REALISTIC: ModelConfig(
            pretrained_model_name="bdsqlsz_controlllite_xl_tile_realistic.safetensors",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CANNY

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ControlNet-LLLite",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet-LLLite model.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            ControlNetLLLite: The loaded model instance.
        """
        filename = self._variant_config.pretrained_model_name
        self.model = load_controlnet_lllite(REPO_ID, filename)

        if dtype_override is not None:
            self.model = self.model.to(dtype_override)

        return self.model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet-LLLite model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            list: Input tensors for the model:
                - x (torch.Tensor): Dummy conditioning input
        """
        if self.model is None:
            self.load_model(dtype_override=dtype_override)

        x = create_dummy_input(self.model)

        if dtype_override:
            x = x.to(dtype_override)

        return [x]
