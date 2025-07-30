# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
UNet torch.hub model loader implementation
"""
import torch
from PIL import Image
from torchvision import transforms
from typing import Optional
from ....tools.utils import get_file
import numpy as np

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available UNet torch.hub model variants."""

    BRAIN_SEG = "brain_segmentation"


class ModelLoader(ForgeModel):
    """UNet torch.hub model loader implementation for image segmentation tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BRAIN_SEG: ModelConfig(
            pretrained_model_name="mateuszbuda/brain-segmentation-pytorch",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BRAIN_SEG

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model="unet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_SEG,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the UNet torch.hub model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The UNet torch.hub model instance.
        """
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        model = torch.hub.load(
            self._variant_config.pretrained_model_name,
            "unet",
            in_channels=3,
            out_channels=1,
            init_features=32,
            pretrained=True,
        )

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the UNet torch.hub model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model inputs' default dtype.
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Input tensor (image data) that can be fed to the model.
        """
        image_file = get_file(
            "https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png"
        )
        input_image = Image.open(str(image_file))
        m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=m, std=s),
            ]
        )
        input_tensor = preprocess(input_image)
        input_batch = torch.stack([input_tensor] * batch_size)
        if dtype_override is not None:
            input_batch = input_batch.to(dtype_override)

        return input_batch
