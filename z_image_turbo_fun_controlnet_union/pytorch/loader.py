# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo-Fun-Controlnet-Union model loader implementation.

Loads the ControlNet Union model for Z-Image-Turbo from alibaba-pai.
This ControlNet supports multiple control conditions including Canny,
Depth, Pose, MLSD, HED, Scribble, and Gray.

Available variants:
- CONTROLNET_UNION_2_1_8STEPS: 8-step distilled ControlNet Union (recommended)
- CONTROLNET_UNION_2_1_LITE_2602_8STEPS: Lite version with fewer control layers
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
from .src.model_utils import (
    create_dummy_control_image,
    load_controlnet_state_dict,
)


class ModelVariant(StrEnum):
    """Available Z-Image-Turbo-Fun-Controlnet-Union model variants."""

    CONTROLNET_UNION_2_1_8STEPS = "ControlNet_Union_2.1_8steps"
    CONTROLNET_UNION_2_1_LITE_2602_8STEPS = "ControlNet_Union_2.1_Lite_2602_8steps"


# Mapping from variant to safetensors filename
_VARIANT_FILENAMES = {
    ModelVariant.CONTROLNET_UNION_2_1_8STEPS: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-8steps.safetensors",
    ModelVariant.CONTROLNET_UNION_2_1_LITE_2602_8STEPS: "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors",
}


class ModelLoader(ForgeModel):
    """Z-Image-Turbo-Fun-Controlnet-Union model loader implementation."""

    _VARIANTS = {
        ModelVariant.CONTROLNET_UNION_2_1_8STEPS: ModelConfig(
            pretrained_model_name="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        ),
        ModelVariant.CONTROLNET_UNION_2_1_LITE_2602_8STEPS: ModelConfig(
            pretrained_model_name="alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union-2.1",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.CONTROLNET_UNION_2_1_8STEPS

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._state_dict = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Z-Image-Turbo-Fun-Controlnet-Union",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CONDITIONAL_GENERATION,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the ControlNet Union state dict.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            dict: The ControlNet state dict containing model weights.
        """
        filename = _VARIANT_FILENAMES[self._variant]
        self._state_dict = load_controlnet_state_dict(filename)

        if dtype_override is not None:
            self._state_dict = {
                k: v.to(dtype_override) for k, v in self._state_dict.items()
            }

        return self._state_dict

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the ControlNet Union model.

        Creates a dummy control conditioning image suitable for any of the
        supported control types (Canny, Depth, Pose, etc.).

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            torch.Tensor: A dummy control image tensor of shape (1, 3, 512, 512).
        """
        control_image = create_dummy_control_image()

        if dtype_override is not None:
            control_image = control_image.to(dtype_override)

        return control_image
