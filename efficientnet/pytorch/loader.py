# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
EfficientNet model loader implementation
"""

import torch
from typing import Optional
from dataclasses import dataclass
from PIL import Image

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
import torchvision.models as models
from torchvision.models._api import WeightsEnum
from ...tools.utils import get_file, print_compiled_model_results, get_state_dict
from torchvision import transforms


@dataclass
class EfficientNetConfig(ModelConfig):
    """Configuration specific to EfficientNet models"""

    model_function: str = None
    weights_class: str = None


class ModelVariant(StrEnum):
    """Available EfficientNet model variants."""

    B0 = "efficientnet_b0"
    B1 = "efficientnet_b1"
    B2 = "efficientnet_b2"
    B3 = "efficientnet_b3"
    B4 = "efficientnet_b4"
    B5 = "efficientnet_b5"
    B6 = "efficientnet_b6"
    B7 = "efficientnet_b7"


class ModelLoader(ForgeModel):
    """EfficientNet model loader implementation."""

    # Static dataclass instances for each variant
    B0_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b0",
        model_function="efficientnet_b0",
        weights_class="EfficientNet_B0_Weights",
    )

    B1_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b1",
        model_function="efficientnet_b1",
        weights_class="EfficientNet_B1_Weights",
    )

    B2_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b2",
        model_function="efficientnet_b2",
        weights_class="EfficientNet_B2_Weights",
    )

    B3_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b3",
        model_function="efficientnet_b3",
        weights_class="EfficientNet_B3_Weights",
    )

    B4_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b4",
        model_function="efficientnet_b4",
        weights_class="EfficientNet_B4_Weights",
    )

    B5_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b5",
        model_function="efficientnet_b5",
        weights_class="EfficientNet_B5_Weights",
    )

    B6_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b6",
        model_function="efficientnet_b6",
        weights_class="EfficientNet_B6_Weights",
    )

    B7_CONFIG = EfficientNetConfig(
        pretrained_model_name="efficientnet_b7",
        model_function="efficientnet_b7",
        weights_class="EfficientNet_B7_Weights",
    )

    # Dictionary using the static dataclass instances (for compatibility with existing tests)
    _VARIANTS = {
        ModelVariant.B0: B0_CONFIG,
        ModelVariant.B1: B1_CONFIG,
        ModelVariant.B2: B2_CONFIG,
        ModelVariant.B3: B3_CONFIG,
        ModelVariant.B4: B4_CONFIG,
        ModelVariant.B5: B5_CONFIG,
        ModelVariant.B6: B6_CONFIG,
        ModelVariant.B7: B7_CONFIG,
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.B0

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
            model="efficientnet",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the EfficientNet model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The EfficientNet model instance.
        """
        # Setup state dict function
        WeightsEnum.get_state_dict = get_state_dict

        # Get model function and weights class from dataclass config
        model_fn = getattr(models, self._variant_config.model_function)
        weights_class = getattr(models, self._variant_config.weights_class)

        # Load model with appropriate weights
        model = model_fn(weights=weights_class.IMAGENET1K_V1)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the EfficientNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for EfficientNet.
        """
        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file)

        # Preprocess image
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        inputs = preprocess(image).unsqueeze(0)

        # Replicate tensors for batch size
        inputs = inputs.repeat_interleave(batch_size, dim=0)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        """Print classification results.

        Args:
            compiled_model_out: Output from the compiled model
        """
        print_compiled_model_results(compiled_model_out)
