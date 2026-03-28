# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AnyPose model loader implementation
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
    load_anypose_pipe,
    create_dummy_images,
    anypose_preprocessing,
)


class ModelVariant(StrEnum):
    """Available AnyPose model variants."""

    ANYPOSE_QWEN_2511 = "AnyPose_Qwen_2511"


class ModelLoader(ForgeModel):
    """AnyPose model loader implementation.

    AnyPose is a LoRA adapter for Qwen/Qwen-Image-Edit-2511 that transfers
    poses from one image to another while preserving the character's appearance
    and background.
    """

    _VARIANTS = {
        ModelVariant.ANYPOSE_QWEN_2511: ModelConfig(
            pretrained_model_name="lilylilith/AnyPose",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.ANYPOSE_QWEN_2511

    prompt = (
        "Make the person in image 1 do the exact same pose of the person in image 2. "
        "Changing the style and background of the image of the person in image 1 is "
        "undesirable, so don't do it. The new pose should be pixel accurate to the pose "
        "we are trying to copy."
    )
    base_model = "Qwen/Qwen-Image-Edit-2511"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.pipeline = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="AnyPose",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMG_TO_IMG,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the AnyPose pipeline.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.

        Returns:
            QwenImageEditPlusPipeline: The pipeline instance with LoRA adapters.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        self.pipeline = load_anypose_pipe(self.base_model, pretrained_model_name)

        if dtype_override is not None:
            self.pipeline = self.pipeline.to(dtype_override)

        return self.pipeline

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the AnyPose model.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.

        Returns:
            dict: Input dictionary for the pipeline containing images and prompt.
        """
        if self.pipeline is None:
            self.load_model(dtype_override=dtype_override)

        character_image, pose_image = create_dummy_images()

        inputs = anypose_preprocessing(
            self.pipeline, self.prompt, character_image, pose_image
        )

        return inputs
