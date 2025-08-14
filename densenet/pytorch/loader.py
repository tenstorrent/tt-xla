# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Densenet model loader implementation
"""

import torch
from typing import Optional
from PIL import Image
from torchvision import transforms
from dataclasses import dataclass
import os

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

import torchxrayvision as xrv
import skimage
import torchvision
from .src.utils import op_norm


@dataclass
class DenseNetConfig(ModelConfig):
    """Configuration specific to DenseNet models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available DenseNet model variants."""

    # Torchvision variants
    DENSENET121 = "densenet121"
    DENSENET161 = "densenet161"
    DENSENET169 = "densenet169"
    DENSENET201 = "densenet201"

    # X-ray variants
    DENSENET121_XRAY = "densenet121_xray"


class ModelLoader(ForgeModel):
    """DenseNet model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # Torchvision variants
        ModelVariant.DENSENET121: DenseNetConfig(
            pretrained_model_name="densenet121",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DENSENET161: DenseNetConfig(
            pretrained_model_name="densenet161",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DENSENET169: DenseNetConfig(
            pretrained_model_name="densenet169",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.DENSENET201: DenseNetConfig(
            pretrained_model_name="densenet201",
            source=ModelSource.TORCH_HUB,
        ),
        # X-ray variants
        ModelVariant.DENSENET121_XRAY: DenseNetConfig(
            pretrained_model_name="densenet121-res224-all",
            source=ModelSource.TORCH_XRAY_VISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DENSENET121

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._cached_model = None

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

        # Get source from variant config
        source = cls._VARIANTS[variant].source

        return ModelInfo(
            model="densenet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained DenseNet model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The DenseNet model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCH_XRAY_VISION:
            # Load X-ray model using torchxrayvision
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
            model = xrv.models.get_model(model_name)
        else:
            # Load model from torch hub
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", model_name, pretrained=True
            )

        model.eval()

        self._cached_model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for DenseNet model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for DenseNet.
        """
        source = self._variant_config.source

        if source == ModelSource.TORCH_XRAY_VISION:
            # Use X-ray specific preprocessing
            img_path = get_file(
                "https://huggingface.co/spaces/torchxrayvision/torchxrayvision-classifier/resolve/main/16747_3_1.jpg"
            )
            img = skimage.io.imread(str(img_path))
            img = xrv.datasets.normalize(img, 255)
            # Check that images are 2D arrays
            if len(img.shape) > 2:
                img = img[:, :, 0]
            if len(img.shape) < 2:
                print("error, dimension lower than 2 for image")
            # Add color channel
            img = img[None, :, :]
            transform = torchvision.transforms.Compose(
                [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)]
            )
            img = transform(img)
            inputs = torch.from_numpy(img).unsqueeze(0)
        else:
            # Standard torchvision preprocessing
            image_file = get_file(
                "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
            )
            image = Image.open(image_file).convert("RGB")

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

    def post_process(self, co_out):
        """
        Post-processes the compiled model output based on the model's source.

        Args:
            co_out : Output from the compiled model.
        """
        source = self._variant_config.source

        if source == ModelSource.TORCH_XRAY_VISION:
            op_threshs = None
            op_threshs = self._cached_model.op_threshs
            op_norm(co_out[0].to(torch.float32), op_threshs.to(torch.float32))
        else:
            print_compiled_model_results(co_out)
