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
from ...tools.utils import (
    VisionPreprocessor,
    VisionPostprocessor,
)


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

    # ImageNet preprocessing constants
    # These values are standard ImageNet preprocessing parameters used by torchvision
    # and are consistent across most ImageNet-trained models:
    # - 256: Resize size (shortest side) - standard preprocessing step before center crop
    # - 224: Final input size after center crop - standard input size for ImageNet models
    IMAGENET_RESIZE_SIZE = 256
    IMAGENET_INPUT_SIZE = 224

    # ImageNet dataset normalization statistics
    # These are the channel-wise mean and standard deviation values computed from
    # the ImageNet training dataset. These statistics are used to normalize input
    # images to match the distribution the models were trained on.
    # Source: Standard ImageNet normalization values used by torchvision
    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB channel means
    IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB channel standard deviations

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
        self.model = None
        self._preprocessor = None
        self._postprocessor = None

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

        # Store model for potential use in input preprocessing and postprocessing
        self.model = model

        # Update preprocessor with cached model (for TIMM models)
        if self._preprocessor is not None:
            self._preprocessor.set_cached_model(model)

        # Update postprocessor with model instance (for HuggingFace models)
        if self._postprocessor is not None:
            self._postprocessor.set_model_instance(model)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def input_preprocess(self, dtype_override=None, batch_size=1, image=None):
        """Preprocess input image(s) and return model-ready input tensor.

        Args:
            dtype_override: Optional torch.dtype override (default: float32).
            batch_size: Batch size (ignored if image is a list).
            image: PIL Image, URL string, tensor, list of images/URLs, or None (uses default COCO image).

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if self._preprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            # For TORCH_HUB, use CUSTOM with standard ImageNet preprocessing
            if source == ModelSource.TORCH_HUB:

                def custom_preprocess_fn(img: Image.Image) -> torch.Tensor:
                    preprocess = transforms.Compose(
                        [
                            transforms.Resize(ModelLoader.IMAGENET_RESIZE_SIZE),
                            transforms.CenterCrop(ModelLoader.IMAGENET_INPUT_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize(
                                mean=ModelLoader.IMAGENET_MEAN,
                                std=ModelLoader.IMAGENET_STD,
                            ),
                        ]
                    )
                    return preprocess(img)

                self._preprocessor = VisionPreprocessor(
                    model_source=ModelSource.CUSTOM,
                    model_name=model_name,
                    custom_preprocess_fn=custom_preprocess_fn,
                )
            else:
                # TIMM source
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
                )

            if hasattr(self, "model") and self.model is not None:
                self._preprocessor.set_cached_model(self.model)

        model_for_config = None
        if self._variant_config.source == ModelSource.TIMM:
            if hasattr(self, "model") and self.model is not None:
                model_for_config = self.model

        return self._preprocessor.preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
            model_for_config=model_for_config,
        )

    def load_inputs(self, dtype_override=None, batch_size=1, image=None):
        """Load and return sample inputs for the model.
        Args:
            dtype_override: Optional torch.dtype override.
            batch_size: Batch size (default: 1).
            image: Optional input image.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        return self.input_preprocess(
            image=image,
            dtype_override=dtype_override,
            batch_size=batch_size,
        )

    def output_postprocess(self, output):
        """Post-process model outputs.

        Args:
            output: Model output tensor.

        Returns:
            dict: Prediction dictionary with top-k results.
        """
        if self._postprocessor is None:
            model_name = self._variant_config.pretrained_model_name
            source = self._variant_config.source

            # For TORCH_HUB, use TORCHVISION postprocessing (same ImageNet labels)
            postprocess_source = (
                ModelSource.TORCHVISION if source == ModelSource.TORCH_HUB else source
            )

            self._postprocessor = VisionPostprocessor(
                model_source=postprocess_source,
                model_name=model_name,
                model_instance=self.model,
            )

        return self._postprocessor.postprocess(output, top_k=1, return_dict=True)
