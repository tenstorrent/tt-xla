# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlexNet model loader implementation
"""

import torch
from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

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
from ...tools.utils import get_file, print_compiled_model_results


@dataclass
class AlexNetConfig(ModelConfig):
    """Configuration specific to AlexNet models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available AlexNet model variants across different sources."""

    ALEXNET_TORCH_HUB = "alexnet"

    ALEXNET_OSMR_B = "alexnetb"


class ModelLoader(ForgeModel):
    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="alexnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    """Loads AlexNet model and sample input."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torch Hub / Torchvision
        ModelVariant.ALEXNET_TORCH_HUB: AlexNetConfig(
            pretrained_model_name="alexnet",
            source=ModelSource.TORCH_HUB,
        ),
        # OSMR variant
        ModelVariant.ALEXNET_OSMR_B: AlexNetConfig(
            pretrained_model_name="alexnetb",
            source=ModelSource.OSMR,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.ALEXNET_TORCH_HUB

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None

    def load_model(self, dtype_override=None):
        """Load pretrained AlexNet model for this instance's variant."""

        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.OSMR:
            model = ptcv_get_model(model_name, pretrained=True)
        else:
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", model_name, pretrained=True
            )

        model.eval()

        # Cache model for use in load_inputs (to avoid reloading)
        self._cached_model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for AlexNet model"""

        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file).convert("RGB")
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

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
