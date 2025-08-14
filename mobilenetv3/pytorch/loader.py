# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobilenetV3 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import torch

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
class MobileNetV3Config(ModelConfig):
    """Configuration specific to MobileNetV3 models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available MobileNetV3 model variants."""

    # TORCH_HUB variants
    MOBILENET_V3_LARGE = "mobilenet_v3_large"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"

    # TIMM variants
    MOBILENET_V3_LARGE_100_TIMM = "mobilenetv3_large_100"
    MOBILENET_V3_SMALL_100_TIMM = "mobilenetv3_small_100"


class ModelLoader(ForgeModel):
    """MobileNetV3 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # TORCH_HUB variants
        ModelVariant.MOBILENET_V3_LARGE: MobileNetV3Config(
            pretrained_model_name="mobilenet_v3_large",
            source=ModelSource.TORCH_HUB,
        ),
        ModelVariant.MOBILENET_V3_SMALL: MobileNetV3Config(
            pretrained_model_name="mobilenet_v3_small",
            source=ModelSource.TORCH_HUB,
        ),
        # TIMM variants
        ModelVariant.MOBILENET_V3_LARGE_100_TIMM: MobileNetV3Config(
            pretrained_model_name="hf_hub:timm/mobilenetv3_large_100.ra_in1k",
            source=ModelSource.TIMM,
        ),
        ModelVariant.MOBILENET_V3_SMALL_100_TIMM: MobileNetV3Config(
            pretrained_model_name="hf_hub:timm/mobilenetv3_small_100.lamb_in1k",
            source=ModelSource.TIMM,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOBILENET_V3_LARGE

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
            model="mobilenetv3",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained MobileNetV3 model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MobileNetV3 model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCH_HUB:
            # Load model using torch hub
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", model_name, pretrained=True
            )
        elif source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)

        model.eval()

        # Cache model for use in load_inputs (to avoid reloading)
        self._cached_model = model

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Prepare sample input for MobileNetV3 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for MobileNetV3.
        """
        source = self._variant_config.source

        # Get the Image
        image_file = get_file(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
        )
        image = Image.open(image_file).convert("RGB")

        if source == ModelSource.TIMM:

            # Use cached model if available, otherwise load it
            if hasattr(self, "_cached_model") and self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                model_for_config = self.load_model(dtype_override)

            # Preprocess image using model's data config
            data_config = resolve_data_config({}, model=model_for_config)
            timm_transforms = create_transform(**data_config)
            inputs = timm_transforms(image).unsqueeze(0)

        else:

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

    def print_cls_results(self, co_out):
        """Print classification results.

        Args:
            co_out: Output from the compiled model
        """
        print_compiled_model_results(co_out)
