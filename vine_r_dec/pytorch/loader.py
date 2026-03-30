# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
VINE-R-Dec model loader implementation for image watermark decoding.
"""
import torch
from PIL import Image
from torchvision import transforms
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
from ...tools.utils import get_file
from .src.custom_convnext import CustomConvNeXt


class ModelVariant(StrEnum):
    """Available VINE-R-Dec model variants."""

    VINE_R_DEC = "R_Dec"


class ModelLoader(ForgeModel):
    """VINE-R-Dec model loader implementation for image watermark decoding tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.VINE_R_DEC: ModelConfig(
            pretrained_model_name="Shilin-LU/VINE-R-Dec",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.VINE_R_DEC

    # Image preprocessing parameters
    image_size = (256, 256)

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.transform_image = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        return ModelInfo(
            model="VINE",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _setup_transforms(self):
        self.transform_image = transforms.Compose(
            [
                transforms.Resize(
                    self.image_size, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
            ]
        )
        return self.transform_image

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = CustomConvNeXt.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()

        if self.transform_image is None:
            self._setup_transforms()

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.transform_image is None:
            self._setup_transforms()

        # Load a sample image
        image_file = get_file(
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
        )
        image = Image.open(image_file).convert("RGB")

        inputs = self.transform_image(image).unsqueeze(0)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        if batch_size > 1:
            inputs = inputs.repeat(batch_size, 1, 1, 1)

        return inputs
