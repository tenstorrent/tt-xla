# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SSDLite320 MobileNetV3 model loader implementation
"""

import torch
from typing import Optional
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel

from PIL import Image
from ...tools.utils import get_file
from torchvision import transforms
import torchvision.models as models


class ModelVariant(StrEnum):
    """Available SSDLite320 MobileNetV3 model variants."""

    SSDLITE320_MOBILENET_V3_LARGE = "ssdlite320_mobilenet_v3_large"


class ModelLoader(ForgeModel):
    """SSDLite320 MobileNetV3 model loader implementation."""

    _VARIANTS = {
        ModelVariant.SSDLITE320_MOBILENET_V3_LARGE: ModelConfig(
            pretrained_model_name="ssdlite320_mobilenet_v3_large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.SSDLITE320_MOBILENET_V3_LARGE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ssdlite320_mobilenetv3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_OBJECT_DET,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the SSDLite320 MobileNetV3 model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).

        Returns:
            torch.nn.Module: The SSDLite320 MobileNetV3 model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name

        # Load model using torchvision with specific weights
        if model_name == "ssdlite320_mobilenet_v3_large":
            weights = models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            model = models.detection.ssdlite320_mobilenet_v3_large(weights=weights)
        else:
            raise ValueError(f"Unsupported model variant: {model_name}")

        model.eval()

        if dtype_override is not None:
            print("NOTE: dtype_override ignored - batched_nms lacks BFloat16 support")
        #     model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the SSDLite320 MobileNetV3 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           NOTE: This parameter is currently ignored (model always uses float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for SSDLite320 MobileNetV3.
        """
        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file)

        # Preprocess image for SSD models
        preprocess = transforms.Compose(
            [
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        inputs = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        if dtype_override is not None:
            print("NOTE: dtype_override ignored - batched_nms lacks BFloat16 support")
            # inputs = inputs.to(dtype_override)

        return inputs
