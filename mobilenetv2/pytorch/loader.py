# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobilenetV2 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
from PIL import Image
from torchvision import transforms
import torch
import timm
from transformers import (
    AutoModelForImageClassification,
    MobileNetV2ForSemanticSegmentation,
)

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
    print_compiled_model_results,
)
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


@dataclass
class MobileNetV2Config(ModelConfig):
    """Configuration specific to MobileNetV2 models"""

    source: ModelSource
    high_res_size: tuple = (
        None  # None means use default size, otherwise (width, height)
    )


class ModelVariant(StrEnum):
    """Available MobileNetV2 model variants."""

    # TORCH_HUB variants
    MOBILENET_V2_TORCH_HUB = "mobilenet_v2"

    # HuggingFace variants
    MOBILENET_V2_035_96_HF = "google/mobilenet_v2_0.35_96"
    MOBILENET_V2_075_160_HF = "google/mobilenet_v2_0.75_160"
    MOBILENET_V2_100_224_HF = "google/mobilenet_v2_1.0_224"
    DEEPLABV3_MOBILENET_V2_HF = "google/deeplabv3_mobilenet_v2_1.0_513"

    # TIMM variants
    MOBILENET_V2_100_TIMM = "mobilenetv2_100"

    # TORCHVISION variants
    MOBILENET_V2_TORCHVISION = "mobilenet_v2_torchvision"


class ModelLoader(ForgeModel):
    """MobileNetV2 model loader implementation."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        # TORCH_HUB variants
        ModelVariant.MOBILENET_V2_TORCH_HUB: MobileNetV2Config(
            pretrained_model_name="mobilenet_v2",
            source=ModelSource.TORCH_HUB,
        ),
        # HuggingFace variants
        ModelVariant.MOBILENET_V2_035_96_HF: MobileNetV2Config(
            pretrained_model_name="google/mobilenet_v2_0.35_96",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.MOBILENET_V2_075_160_HF: MobileNetV2Config(
            pretrained_model_name="google/mobilenet_v2_0.75_160",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.MOBILENET_V2_100_224_HF: MobileNetV2Config(
            pretrained_model_name="google/mobilenet_v2_1.0_224",
            source=ModelSource.HUGGING_FACE,
        ),
        ModelVariant.DEEPLABV3_MOBILENET_V2_HF: MobileNetV2Config(
            pretrained_model_name="google/deeplabv3_mobilenet_v2_1.0_513",
            source=ModelSource.HUGGING_FACE,
        ),
        # TIMM variants
        ModelVariant.MOBILENET_V2_100_TIMM: MobileNetV2Config(
            pretrained_model_name="mobilenetv2_100",
            source=ModelSource.TIMM,
        ),
        # TORCHVISION variants
        ModelVariant.MOBILENET_V2_TORCHVISION: MobileNetV2Config(
            pretrained_model_name="mobilenet_v2",
            source=ModelSource.TORCHVISION,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MOBILENET_V2_TORCH_HUB

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

        if variant in [ModelVariant.MOBILENET_V2_TORCH_HUB]:
            group = ModelGroup.RED
        else:
            group = ModelGroup.GENERALITY

        return ModelInfo(
            model="mobilenetv2",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load pretrained MobileNetV2 model for this instance's variant.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The MobileNetV2 model instance.
        """
        # Get the pretrained model name from the instance's variant config
        model_name = self._variant_config.pretrained_model_name
        source = self._variant_config.source

        if source == ModelSource.TORCH_HUB:
            # Load model using torch hub
            model = torch.hub.load(
                "pytorch/vision:v0.10.0", model_name, pretrained=True
            )

        elif source == ModelSource.HUGGING_FACE:
            # Load model using HuggingFace transformers
            if "deeplabv3" in model_name.lower():
                # For DeepLabV3 models, use semantic segmentation model
                model = MobileNetV2ForSemanticSegmentation.from_pretrained(model_name)
            else:
                # For standard classification models
                model = AutoModelForImageClassification.from_pretrained(model_name)

        elif source == ModelSource.TIMM:
            # Load model using timm
            model = timm.create_model(model_name, pretrained=True)

        elif source == ModelSource.TORCHVISION:
            # Load model using torchvision
            model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

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
            high_res_size = self._variant_config.high_res_size

            def weight_class_name_fn(name: str) -> str:
                # Handle mobilenet_v2 -> MobileNet_V2_Weights
                if name == "mobilenet_v2":
                    return "MobileNet_V2_Weights"
                # Handle other potential naming patterns
                return (
                    name.replace("mobilenet", "MobileNet").replace("_", "") + "_Weights"
                )

            # For TORCH_HUB, use CUSTOM with standard ImageNet preprocessing
            if source == ModelSource.TORCH_HUB:

                def custom_preprocess_fn(img: Image.Image) -> torch.Tensor:
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
                    return preprocess(img)

                self._preprocessor = VisionPreprocessor(
                    model_source=ModelSource.CUSTOM,
                    model_name=model_name,
                    high_res_size=high_res_size,
                    custom_preprocess_fn=custom_preprocess_fn,
                )
            else:
                self._preprocessor = VisionPreprocessor(
                    model_source=source,
                    model_name=model_name,
                    high_res_size=high_res_size,
                    weight_class_name_fn=(
                        weight_class_name_fn
                        if source == ModelSource.TORCHVISION
                        else None
                    ),
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
        """Load and return sample inputs (backward compatibility wrapper for input_preprocess).

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

    def output_postprocess(
        self,
        output=None,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
    ):
        """Post-process model outputs.

        Args:
            output: Model output tensor (returns dict if provided).
            co_out: Compiled model outputs (legacy, prints results).
            framework_model: Original framework model (legacy).
            compiled_model: Compiled model (legacy).
            inputs: Input images (legacy).
            dtype_override: Optional dtype override (legacy).

        Returns:
            dict or None: Prediction dict if output provided, else None (prints results).
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

        # New usage: return dict from output tensor
        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)

        # Legacy usage: print results (backward compatibility)
        self._postprocessor.print_results(
            co_out=co_out,
            framework_model=framework_model,
            compiled_model=compiled_model,
            inputs=inputs,
            dtype_override=dtype_override,
        )
        return None

    def print_cls_results(self, compiled_model_out):
        """Legacy method for backward compatibility."""
        print_compiled_model_results(compiled_model_out)
