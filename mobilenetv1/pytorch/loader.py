# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobilenetV1 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

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
from .src.utils import MobileNetV1
from transformers import AutoModelForImageClassification
from transformers import AutoImageProcessor
from datasets import load_dataset


@dataclass
class MobileNetV1Config(ModelConfig):
    """Configuration specific to MobileNetV1 models"""

    source: ModelSource


class ModelVariant(StrEnum):
    """Available MobileNetV1 model variants."""

    # GitHub variants
    MOBILENET_V1_GITHUB = "mobilenet_v1"

    # HuggingFace variants
    MOBILENET_V1_075_192_HF = "google/mobilenet_v1_0.75_192"
    MOBILENET_V1_100_224_HF = "google/mobilenet_v1_1.0_224"

    # TIMM variants
    MOBILENET_V1_100_TIMM = "mobilenetv1_100.ra4_e3600_r224_in1k"


class ModelLoader(ForgeModel):
    """MobileNetV1 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # GitHub variants
        ModelVariant.MOBILENET_V1_GITHUB: MobileNetV1Config(
            pretrained_model_name="mobilenet_v1",
            source=ModelSource.GITHUB,
        ),
        # HuggingFace variants
        ModelVariant.MOBILENET_V1_075_192_HF: MobileNetV1Config(
            pretrained_model_name="google/mobilenet_v1_0.75_192",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.MOBILENET_V1_100_224_HF: MobileNetV1Config(
            pretrained_model_name="google/mobilenet_v1_1.0_224",
            source=ModelSource.HUGGING_FACE,
        ),
        # TIMM variants
        ModelVariant.MOBILENET_V1_100_TIMM: MobileNetV1Config(
            pretrained_model_name="mobilenetv1_100.ra4_e3600_r224_in1k",
            source=ModelSource.TIMM,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOBILENET_V1_GITHUB

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
            model="mobilenetv1",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained MobileNetV1 model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MobileNetV1 model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.GITHUB:
            # Load model using GitHub source implementation
            model = MobileNetV1(9)
        elif source == ModelSource.HUGGING_FACE:
            # Load model using HuggingFace transformers
            model = AutoModelForImageClassification.from_pretrained(model_name)
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
        """Prepare sample input for MobileNetV1 model with this instance's variant settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for MobileNetV1.
        """
        source = self._variant_config.source

        if source == ModelSource.HUGGING_FACE:
            model_name = self._variant_config.pretrained_model_name
            preprocessor = AutoImageProcessor.from_pretrained(model_name)
            dataset = load_dataset("imagenet-1k", split="validation", streaming=True)
            image = next(iter(dataset.skip(10)))["image"]
            input_dict = preprocessor(images=image, return_tensors="pt")
            inputs = input_dict.pixel_values
        elif source == ModelSource.TIMM:
            image_file = get_file(
                "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
            )
            image = Image.open(image_file).convert("RGB")

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
            # Standard preprocessing for GitHub and other sources
            image_file = get_file(
                "https://images.rawpixel.com/image_1300/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIyLTA1L3BkMTA2LTA0Ny1jaGltXzEuanBn.jpg"
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

    def print_cls_results(self, compiled_model_out):
        print_compiled_model_results(compiled_model_out)
