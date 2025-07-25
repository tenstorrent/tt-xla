# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ResNeXt model loader implementation
"""

import torch
from typing import Optional
from dataclasses import dataclass
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
from ...tools.utils import get_file, print_compiled_model_results
from torchvision import transforms


@dataclass
class ResNeXtConfig(ModelConfig):
    """Configuration specific to ResNeXt models"""

    hub_source: str


class ModelVariant(StrEnum):
    """Available ResNeXt model variants."""

    RESNEXT50_32X4D = "resnext50_32x4d"
    RESNEXT101_32X8D = "resnext101_32x8d"
    RESNEXT101_32X8D_WSL = "resnext101_32x8d_wsl"


class ModelLoader(ForgeModel):
    """ResNeXt model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.RESNEXT50_32X4D: ResNeXtConfig(
            pretrained_model_name="resnext50_32x4d",
            hub_source="pytorch/vision:v0.10.0",
        ),
        ModelVariant.RESNEXT101_32X8D: ResNeXtConfig(
            pretrained_model_name="resnext101_32x8d",
            hub_source="pytorch/vision:v0.10.0",
        ),
        ModelVariant.RESNEXT101_32X8D_WSL: ResNeXtConfig(
            pretrained_model_name="resnext101_32x8d_wsl",
            hub_source="facebookresearch/WSL-Images",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.RESNEXT50_32X4D

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
            model="resnext",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCH_HUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the ResNeXt model instance for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The ResNeXt model instance.
        """
        # Get the pretrained model name and hub source from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        hub_source = self._variant_config.hub_source

        # Load model using torch.hub
        model = torch.hub.load(hub_source, model_name)
        model.eval()

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the ResNeXt model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for ResNeXt.
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
