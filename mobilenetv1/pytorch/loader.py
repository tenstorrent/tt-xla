# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MobilenetV1 model loader implementation
"""

from typing import Optional
from dataclasses import dataclass
import timm
from transformers import AutoModelForImageClassification

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
from ...tools.utils import VisionPreprocessor, VisionPostprocessor
from datasets import load_dataset
from .src.utils import MobileNetV1


@dataclass
class MobileNetV1Config(ModelConfig):
    """Configuration specific to MobileNetV1 models"""

    source: ModelSource
    group: ModelGroup = ModelGroup.GENERALITY


class ModelVariant(StrEnum):
    """Available MobileNetV1 model variants."""

    # GitHub variants
    MOBILENET_V1_GITHUB = "Mobilenet_v1"

    # HuggingFace variants
    MOBILENET_V1_075_192_HF = "Mobilenet_v1_0.75_192"
    MOBILENET_V1_100_224_HF = "Mobilenet_v1_1.0_224"

    # Optimum Intel variants
    MOBILENET_V1_075_192_OPTIMUM = "Optimum_Mobilenet_v1_0.75_192"

    # TIMM variants
    MOBILENET_V1_100_TIMM = "100.ra4_E3600_R224_In1k"


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
        # Optimum Intel variants
        ModelVariant.MOBILENET_V1_075_192_OPTIMUM: MobileNetV1Config(
            pretrained_model_name="optimum-intel-internal-testing/mobilenet_v1_0.75_192",
            source=ModelSource.HUGGING_FACE,
            group=ModelGroup.VULCAN,
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

        # Get group from variant config
        group = cls._VARIANTS[variant].group

        return ModelInfo(
            model="MobileNetV1",
            variant=variant,
            group=group,
            task=ModelTask.CV_IMAGE_CLS,
            source=source,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
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
            model = AutoModelForImageClassification.from_pretrained(
                model_name, **kwargs
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

            # Handle different sources
            if source == ModelSource.TIMM:
                preprocessor_source = ModelSource.TIMM
                preprocessor_model_name = model_name
            elif source == ModelSource.HUGGING_FACE:
                preprocessor_source = ModelSource.HUGGING_FACE
                preprocessor_model_name = model_name
            elif source == ModelSource.GITHUB:
                # GitHub models use standard ImageNet preprocessing
                preprocessor_source = ModelSource.CUSTOM
                from torchvision import transforms

                def custom_preprocess_fn(img):
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

            else:
                raise ValueError(f"Unsupported source for preprocessing: {source}")

            # Create preprocessor
            if source == ModelSource.GITHUB:
                self._preprocessor = VisionPreprocessor(
                    model_source=preprocessor_source,
                    model_name=model_name,
                    custom_preprocess_fn=custom_preprocess_fn,
                )
            else:
                self._preprocessor = VisionPreprocessor(
                    model_source=preprocessor_source,
                    model_name=preprocessor_model_name,
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
            image: Optional input image. If None, loads from HuggingFace datasets.

        Returns:
            torch.Tensor: Preprocessed input tensor.
        """
        if image is None:
            dataset = load_dataset("huggingface/cats-image", split="test")
            image = dataset[0]["image"]
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

            # Map sources to postprocessor sources
            if source == ModelSource.TIMM:
                postprocessor_source = ModelSource.TIMM
                postprocessor_model_name = model_name
            elif source == ModelSource.HUGGING_FACE:
                postprocessor_source = ModelSource.HUGGING_FACE
                postprocessor_model_name = model_name
            elif source == ModelSource.GITHUB:
                # GitHub models use ImageNet labels like torchvision
                postprocessor_source = ModelSource.TORCHVISION
                postprocessor_model_name = (
                    "mobilenet_v2"  # Use a standard torchvision name for labels
                )
            else:
                raise ValueError(f"Unsupported source for postprocessing: {source}")

            self._postprocessor = VisionPostprocessor(
                model_source=postprocessor_source,
                model_name=postprocessor_model_name,
                model_instance=self.model,
            )

        # New usage: return dict from output tensor
        if output is not None:
            return self._postprocessor.postprocess(output, top_k=1, return_dict=True)

        # Legacy usage: print results (backward compatibility)
        if co_out is not None:
            from ...tools.utils import print_compiled_model_results

            print_compiled_model_results(co_out)
        return None

    def print_cls_results(self, compiled_model_out):
        """Print classification results (backward compatibility).

        Args:
            compiled_model_out: Output from the compiled model
        """
        from ...tools.utils import print_compiled_model_results

        print_compiled_model_results(compiled_model_out)
